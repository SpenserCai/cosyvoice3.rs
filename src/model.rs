//! CosyVoice3 model implementation

use crate::config::CosyVoice3Config;
use crate::device::PyDevice;
use crate::error::{wrap_candle_err, CosyVoice3Error};
use crate::{SamplingConfig, SynthesisMode};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::cosyvoice::{
    CausalHiFTGenerator, CausalMaskedDiffWithDiT, CosyVoice3LM, DiT,
};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// CosyVoice3 TTS model
#[pyclass]
pub struct CosyVoice3 {
    inner: Mutex<CosyVoice3Inner>,
    config: CosyVoice3Config,
    device: Device,
    dtype: DType,
}

struct CosyVoice3Inner {
    llm: CosyVoice3LM,
    flow_decoder: CausalMaskedDiffWithDiT,
    vocoder: CausalHiFTGenerator,
    tokenizer: Tokenizer,
}

#[pymethods]
impl CosyVoice3 {
    /// Load CosyVoice3 model from a directory
    ///
    /// Args:
    ///     model_dir: Path to the model directory containing weights and config
    ///     device: Device to run the model on ("cpu", "cuda", or "metal")
    ///     use_f16: Whether to use f16 precision (GPU only)
    ///
    /// Returns:
    ///     CosyVoice3 model instance
    #[new]
    #[pyo3(signature = (model_dir, device=None, use_f16=false))]
    fn new(model_dir: &str, device: Option<PyDevice>, use_f16: bool) -> PyResult<Self> {
        let model_path = PathBuf::from(model_dir);

        // Load config
        let config = CosyVoice3Config::from_model_dir(&model_path)?;

        // Setup device
        let py_device = device.unwrap_or_else(PyDevice::best_available);
        let candle_device = py_device.to_candle_device()?;

        // Setup dtype
        let dtype = if use_f16 && py_device != PyDevice::Cpu {
            DType::F16
        } else {
            DType::F32
        };

        // Load tokenizer
        let tokenizer = Self::load_tokenizer(&model_path)?;

        // Load model weights
        let llm_path = model_path.join("llm.safetensors");
        let flow_path = model_path.join("flow.safetensors");
        let hift_path = model_path.join("hift.safetensors");

        let llm_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&llm_path], dtype, &candle_device) }
                .map_err(wrap_candle_err)?;
        let flow_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&flow_path], dtype, &candle_device) }
                .map_err(wrap_candle_err)?;
        let hift_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&hift_path], dtype, &candle_device) }
                .map_err(wrap_candle_err)?;

        // Create LLM
        let llm_config = config.to_llm_config();
        let llm = CosyVoice3LM::new(&llm_config, llm_vb).map_err(wrap_candle_err)?;

        // Create Flow Decoder
        let flow_config = config.to_flow_config();
        let dit = DiT::new(flow_config.dit.clone(), flow_vb.pp("dit")).map_err(wrap_candle_err)?;
        let flow_decoder = CausalMaskedDiffWithDiT::new(
            flow_config.vocab_size,
            flow_config.output_size,
            flow_config.output_size,
            config.spk_embed_dim,
            flow_config.token_mel_ratio,
            flow_config.pre_lookahead_len,
            dit,
            flow_config.cfm.clone(),
            flow_vb,
        )
        .map_err(wrap_candle_err)?;

        // Create Vocoder
        let hift_config = config.to_hift_config();
        let vocoder = CausalHiFTGenerator::new(hift_config, hift_vb).map_err(wrap_candle_err)?;

        let inner = CosyVoice3Inner {
            llm,
            flow_decoder,
            vocoder,
            tokenizer,
        };

        Ok(Self {
            inner: Mutex::new(inner),
            config,
            device: candle_device,
            dtype,
        })
    }

    /// Synthesize speech from text
    ///
    /// Args:
    ///     text: Text to synthesize
    ///     prompt_speech_tokens: Speech tokens from prompt audio
    ///     prompt_mel: Mel spectrogram from prompt audio (shape: [1, T, 80])
    ///     speaker_embedding: Speaker embedding (shape: [1, 192])
    ///     prompt_text: Prompt text for zero-shot mode
    ///     mode: Synthesis mode (zero_shot, cross_lingual, instruct)
    ///     instruct_text: Instruction text for instruct mode
    ///     sampling_config: Sampling configuration
    ///     n_timesteps: Number of CFM sampling steps
    ///
    /// Returns:
    ///     Audio waveform as a list of floats
    #[pyo3(signature = (
        text,
        prompt_speech_tokens,
        prompt_mel,
        speaker_embedding,
        prompt_text="You are a helpful assistant.<|endofprompt|>",
        mode=SynthesisMode::ZeroShot,
        instruct_text=None,
        sampling_config=None,
        n_timesteps=10
    ))]
    #[allow(clippy::too_many_arguments)]
    fn synthesize(
        &self,
        text: &str,
        prompt_speech_tokens: Vec<u32>,
        prompt_mel: Vec<Vec<f32>>,
        speaker_embedding: Vec<f32>,
        prompt_text: &str,
        mode: SynthesisMode,
        instruct_text: Option<&str>,
        sampling_config: Option<SamplingConfig>,
        n_timesteps: usize,
    ) -> PyResult<Vec<f32>> {
        let sampling = sampling_config.unwrap_or_else(|| SamplingConfig::new(25, 0.8, 1.0, 1.0));

        // Convert inputs to tensors
        let prompt_mel_tensor = self.vec2d_to_tensor(&prompt_mel)?;
        let speaker_embedding_tensor = self.vec1d_to_tensor(&speaker_embedding, (1, 192))?;

        let mut inner = self.inner.lock().unwrap();

        // Tokenize text
        let text_tokens = self.tokenize(&inner.tokenizer, text)?;

        // Determine prompt text and LLM speech tokens based on mode
        let (actual_prompt_text, llm_speech_tokens) = match mode {
            SynthesisMode::ZeroShot => (prompt_text, prompt_speech_tokens.clone()),
            SynthesisMode::CrossLingual => ("", vec![]),
            SynthesisMode::Instruct => {
                let instruct = instruct_text
                    .unwrap_or("You are a helpful assistant.<|endofprompt|>");
                (instruct, vec![])
            }
        };

        let prompt_text_tokens = self.tokenize(&inner.tokenizer, actual_prompt_text)?;

        // Create tensors
        let text_tokens_tensor =
            Tensor::from_slice(&text_tokens, (1, text_tokens.len()), &self.device)
                .map_err(wrap_candle_err)?
                .to_dtype(DType::U32)
                .map_err(wrap_candle_err)?;

        let prompt_text_tensor = if prompt_text_tokens.is_empty() {
            Tensor::zeros((1, 0), DType::U32, &self.device).map_err(wrap_candle_err)?
        } else {
            Tensor::from_slice(
                &prompt_text_tokens,
                (1, prompt_text_tokens.len()),
                &self.device,
            )
            .map_err(wrap_candle_err)?
            .to_dtype(DType::U32)
            .map_err(wrap_candle_err)?
        };

        let llm_prompt_speech_tensor = if llm_speech_tokens.is_empty() {
            Tensor::zeros((1, 0), DType::U32, &self.device).map_err(wrap_candle_err)?
        } else {
            Tensor::from_slice(&llm_speech_tokens, (1, llm_speech_tokens.len()), &self.device)
                .map_err(wrap_candle_err)?
                .to_dtype(DType::U32)
                .map_err(wrap_candle_err)?
        };

        let flow_prompt_speech_tensor = Tensor::from_slice(
            &prompt_speech_tokens,
            (1, prompt_speech_tokens.len()),
            &self.device,
        )
        .map_err(wrap_candle_err)?
        .to_dtype(DType::U32)
        .map_err(wrap_candle_err)?;

        // LLM inference
        let candle_sampling = (&sampling).into();
        let speech_tokens = inner
            .llm
            .inference(
                &text_tokens_tensor,
                &prompt_text_tensor,
                &llm_prompt_speech_tensor,
                &candle_sampling,
            )
            .map_err(wrap_candle_err)?;

        if speech_tokens.is_empty() {
            return Err(CosyVoice3Error::Model("LLM generated no speech tokens".to_string()).into());
        }

        // Flow decoder
        let speech_tokens_tensor =
            Tensor::from_slice(&speech_tokens, (1, speech_tokens.len()), &self.device)
                .map_err(wrap_candle_err)?
                .to_dtype(DType::U32)
                .map_err(wrap_candle_err)?;

        let mel = inner
            .flow_decoder
            .inference(
                &speech_tokens_tensor,
                &flow_prompt_speech_tensor,
                &prompt_mel_tensor,
                &speaker_embedding_tensor,
                n_timesteps,
                false,
            )
            .map_err(wrap_candle_err)?;

        let mel = mel
            .to_device(&self.device)
            .map_err(wrap_candle_err)?
            .to_dtype(DType::F32)
            .map_err(wrap_candle_err)?;

        // Vocoder
        let waveform = inner
            .vocoder
            .inference(&mel, true)
            .map_err(wrap_candle_err)?;

        // Extract PCM data
        let pcm = if waveform.dims().len() == 3 {
            waveform
                .squeeze(0)
                .map_err(wrap_candle_err)?
                .squeeze(0)
                .map_err(wrap_candle_err)?
        } else if waveform.dims().len() == 2 {
            waveform.squeeze(0).map_err(wrap_candle_err)?
        } else {
            waveform
        };

        let pcm_data: Vec<f32> = pcm
            .to_dtype(DType::F32)
            .map_err(wrap_candle_err)?
            .to_vec1()
            .map_err(wrap_candle_err)?;

        Ok(pcm_data)
    }



    /// Get the sample rate
    #[getter]
    fn sample_rate(&self) -> usize {
        self.config.sample_rate
    }

    /// Get the configuration
    #[getter]
    fn config(&self) -> CosyVoice3Config {
        self.config.clone()
    }

    /// Check if ONNX feature is compiled
    #[getter]
    fn has_onnx(&self) -> bool {
        cfg!(feature = "onnx")
    }

    fn __repr__(&self) -> String {
        format!(
            "CosyVoice3(sample_rate={}, device={:?}, dtype={:?})",
            self.config.sample_rate, self.device, self.dtype
        )
    }
}

impl CosyVoice3 {
    fn load_tokenizer(model_path: &std::path::Path) -> PyResult<Tokenizer> {
        let tokenizer_path = model_path.join("tokenizer");

        if tokenizer_path.join("tokenizer.json").exists() {
            Tokenizer::from_file(tokenizer_path.join("tokenizer.json")).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load tokenizer: {}", e))
            })
        } else {
            // Build from vocab.json + merges.txt
            let vocab_path = tokenizer_path.join("vocab.json");
            let merges_path = tokenizer_path.join("merges.txt");

            use tokenizers::models::bpe::BPE;
            let bpe = BPE::from_file(
                &vocab_path.to_string_lossy(),
                &merges_path.to_string_lossy(),
            )
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to build tokenizer: {}", e))
            })?;

            Ok(Tokenizer::new(bpe))
        }
    }

    fn tokenize(&self, tokenizer: &Tokenizer, text: &str) -> PyResult<Vec<u32>> {
        let encoding = tokenizer.encode(text, false).map_err(|e| {
            CosyVoice3Error::Tokenizer(format!("Failed to tokenize: {}", e))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    fn vec2d_to_tensor(&self, data: &[Vec<f32>]) -> PyResult<Tensor> {
        if data.is_empty() {
            return Err(CosyVoice3Error::InvalidArgument("Empty mel data".to_string()).into());
        }

        let t_dim = data.len();
        let mel_dim = data[0].len();

        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let tensor = Tensor::from_vec(flat, (1, t_dim, mel_dim), &self.device)
            .map_err(wrap_candle_err)?
            .to_dtype(self.dtype)
            .map_err(wrap_candle_err)?;

        Ok(tensor)
    }

    fn vec1d_to_tensor(&self, data: &[f32], shape: (usize, usize)) -> PyResult<Tensor> {
        let tensor = Tensor::from_slice(data, shape, &self.device)
            .map_err(wrap_candle_err)?
            .to_dtype(self.dtype)
            .map_err(wrap_candle_err)?;
        Ok(tensor)
    }
}
