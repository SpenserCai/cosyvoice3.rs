//! CosyVoice3 model implementation

use crate::audio::{pcm_decode, AudioInput};
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

/// Type alias for prompt features: (speech_tokens, mel_spectrogram, speaker_embedding)
type PromptFeatures = (Vec<u32>, Vec<Vec<f32>>, Vec<f32>);

/// CosyVoice3 TTS model
#[pyclass]
pub struct CosyVoice3 {
    inner: Mutex<CosyVoice3Inner>,
    config: CosyVoice3Config,
    device: Device,
    dtype: DType,
    #[allow(dead_code)]
    model_dir: PathBuf,
    #[cfg(feature = "onnx")]
    frontend: Option<candle_transformers::models::cosyvoice::CosyVoice3Frontend>,
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

        // Load frontend for feature extraction (ONNX only)
        #[cfg(feature = "onnx")]
        let frontend = {
            use candle_transformers::models::cosyvoice::CosyVoice3Frontend;
            CosyVoice3Frontend::load(&model_path, &Device::Cpu).ok()
        };

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
            model_dir: model_path,
            #[cfg(feature = "onnx")]
            frontend,
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

    /// Zero-shot voice cloning inference
    ///
    /// Synthesize speech using a prompt audio for voice cloning.
    /// This is the simplest API - just provide text and a prompt audio file.
    ///
    /// Args:
    ///     text: Text to synthesize
    ///     prompt_text: Prompt text (should match the content of prompt audio)
    ///     prompt_wav: Path to prompt audio file (WAV/MP3/OGG) or list of audio samples
    ///     prompt_wav_sample_rate: Sample rate of prompt audio (only needed if prompt_wav is samples)
    ///     sampling_config: Sampling configuration (optional)
    ///     n_timesteps: Number of CFM sampling steps (default: 10)
    ///
    /// Returns:
    ///     Audio waveform as a list of floats
    ///
    /// Example:
    ///     >>> model = CosyVoice3("model_dir")
    ///     >>> audio = model.inference_zero_shot(
    ///     ...     "Hello, world!",
    ///     ...     "You are a helpful assistant.<|endofprompt|>Hi there.",
    ///     ...     "prompt.wav"
    ///     ... )
    #[pyo3(signature = (text, prompt_text, prompt_wav, prompt_wav_sample_rate=None, sampling_config=None, n_timesteps=10))]
    fn inference_zero_shot(
        &self,
        text: &str,
        prompt_text: &str,
        prompt_wav: AudioInput,
        prompt_wav_sample_rate: Option<u32>,
        sampling_config: Option<SamplingConfig>,
        n_timesteps: usize,
    ) -> PyResult<Vec<f32>> {
        let (prompt_speech_tokens, prompt_mel, speaker_embedding) =
            self.extract_prompt_features(prompt_wav, prompt_wav_sample_rate)?;

        self.synthesize(
            text,
            prompt_speech_tokens,
            prompt_mel,
            speaker_embedding,
            prompt_text,
            SynthesisMode::ZeroShot,
            None,
            sampling_config,
            n_timesteps,
        )
    }

    /// Cross-lingual voice cloning inference
    ///
    /// Synthesize speech in a different language while preserving the voice from prompt audio.
    /// The LLM does not receive prompt text or speech tokens, only the flow decoder uses them.
    ///
    /// Args:
    ///     text: Text to synthesize (can include language tags like <|en|>, <|zh|>, etc.)
    ///     prompt_wav: Path to prompt audio file or list of audio samples
    ///     prompt_wav_sample_rate: Sample rate of prompt audio (only needed if prompt_wav is samples)
    ///     sampling_config: Sampling configuration (optional)
    ///     n_timesteps: Number of CFM sampling steps (default: 10)
    ///
    /// Returns:
    ///     Audio waveform as a list of floats
    ///
    /// Example:
    ///     >>> model = CosyVoice3("model_dir")
    ///     >>> audio = model.inference_cross_lingual(
    ///     ...     "<|en|>Hello, this is cross-lingual synthesis.",
    ///     ...     "chinese_prompt.wav"
    ///     ... )
    #[pyo3(signature = (text, prompt_wav, prompt_wav_sample_rate=None, sampling_config=None, n_timesteps=10))]
    fn inference_cross_lingual(
        &self,
        text: &str,
        prompt_wav: AudioInput,
        prompt_wav_sample_rate: Option<u32>,
        sampling_config: Option<SamplingConfig>,
        n_timesteps: usize,
    ) -> PyResult<Vec<f32>> {
        let (prompt_speech_tokens, prompt_mel, speaker_embedding) =
            self.extract_prompt_features(prompt_wav, prompt_wav_sample_rate)?;

        self.synthesize(
            text,
            prompt_speech_tokens,
            prompt_mel,
            speaker_embedding,
            "", // Empty prompt text for cross-lingual
            SynthesisMode::CrossLingual,
            None,
            sampling_config,
            n_timesteps,
        )
    }

    /// Instruction-based voice synthesis
    ///
    /// Synthesize speech with specific instructions (e.g., speaking style, dialect).
    /// The instruction text guides the synthesis while the prompt audio provides the voice.
    ///
    /// Args:
    ///     text: Text to synthesize
    ///     instruct_text: Instruction for synthesis style (e.g., "请用广东话表达。<|endofprompt|>")
    ///     prompt_wav: Path to prompt audio file or list of audio samples
    ///     prompt_wav_sample_rate: Sample rate of prompt audio (only needed if prompt_wav is samples)
    ///     sampling_config: Sampling configuration (optional)
    ///     n_timesteps: Number of CFM sampling steps (default: 10)
    ///
    /// Returns:
    ///     Audio waveform as a list of floats
    ///
    /// Example:
    ///     >>> model = CosyVoice3("model_dir")
    ///     >>> audio = model.inference_instruct(
    ///     ...     "你好世界",
    ///     ...     "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
    ///     ...     "prompt.wav"
    ///     ... )
    #[pyo3(signature = (text, instruct_text, prompt_wav, prompt_wav_sample_rate=None, sampling_config=None, n_timesteps=10))]
    fn inference_instruct(
        &self,
        text: &str,
        instruct_text: &str,
        prompt_wav: AudioInput,
        prompt_wav_sample_rate: Option<u32>,
        sampling_config: Option<SamplingConfig>,
        n_timesteps: usize,
    ) -> PyResult<Vec<f32>> {
        let (prompt_speech_tokens, prompt_mel, speaker_embedding) =
            self.extract_prompt_features(prompt_wav, prompt_wav_sample_rate)?;

        self.synthesize(
            text,
            prompt_speech_tokens,
            prompt_mel,
            speaker_embedding,
            instruct_text,
            SynthesisMode::Instruct,
            Some(instruct_text),
            sampling_config,
            n_timesteps,
        )
    }

    /// Load prompt features from a safetensors file
    ///
    /// This is useful when you have pre-extracted features and want to reuse them.
    ///
    /// Args:
    ///     features_path: Path to the safetensors file containing prompt features
    ///
    /// Returns:
    ///     Tuple of (prompt_speech_tokens, prompt_mel, speaker_embedding)
    #[pyo3(signature = (features_path,))]
    fn load_prompt_features(&self, features_path: &str) -> PyResult<PromptFeatures> {
        let features = candle_core::safetensors::load(features_path, &self.device)
            .map_err(wrap_candle_err)?;

        // Load speech tokens
        let tokens_tensor = features
            .get("prompt_speech_tokens")
            .ok_or_else(|| {
                CosyVoice3Error::InvalidArgument(
                    "Missing prompt_speech_tokens in features file".to_string(),
                )
            })?;

        let tokens: Vec<u32> = if tokens_tensor.dtype() == DType::I64 {
            tokens_tensor
                .flatten_all()
                .map_err(wrap_candle_err)?
                .to_vec1::<i64>()
                .map_err(wrap_candle_err)?
                .into_iter()
                .map(|x| x as u32)
                .collect()
        } else {
            tokens_tensor
                .flatten_all()
                .map_err(wrap_candle_err)?
                .to_vec1::<i32>()
                .map_err(wrap_candle_err)?
                .into_iter()
                .map(|x| x as u32)
                .collect()
        };

        // Load mel spectrogram
        let mel_tensor = features.get("prompt_mel").ok_or_else(|| {
            CosyVoice3Error::InvalidArgument("Missing prompt_mel in features file".to_string())
        })?;

        let mel_shape = mel_tensor.dims();
        let t_dim = if mel_shape.len() == 3 {
            mel_shape[1]
        } else {
            mel_shape[0]
        };
        let mel_dim = if mel_shape.len() == 3 {
            mel_shape[2]
        } else {
            mel_shape[1]
        };

        let mel_flat: Vec<f32> = mel_tensor
            .flatten_all()
            .map_err(wrap_candle_err)?
            .to_dtype(DType::F32)
            .map_err(wrap_candle_err)?
            .to_vec1()
            .map_err(wrap_candle_err)?;

        let mel: Vec<Vec<f32>> = mel_flat.chunks(mel_dim).map(|c| c.to_vec()).collect();
        let mel = mel.into_iter().take(t_dim).collect();

        // Load speaker embedding
        let spk_tensor = features.get("speaker_embedding").ok_or_else(|| {
            CosyVoice3Error::InvalidArgument(
                "Missing speaker_embedding in features file".to_string(),
            )
        })?;

        let speaker_embedding: Vec<f32> = spk_tensor
            .flatten_all()
            .map_err(wrap_candle_err)?
            .to_dtype(DType::F32)
            .map_err(wrap_candle_err)?
            .to_vec1()
            .map_err(wrap_candle_err)?;

        Ok((tokens, mel, speaker_embedding))
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
    /// Extract prompt features from audio input
    ///
    /// This method handles both file paths and raw audio samples.
    /// When ONNX feature is enabled, it uses the frontend to extract features.
    /// Otherwise, it returns an error.
    fn extract_prompt_features(
        &self,
        prompt_wav: AudioInput,
        sample_rate: Option<u32>,
    ) -> PyResult<PromptFeatures> {
        // Load audio data
        let (audio_data, audio_sample_rate) = match prompt_wav {
            AudioInput::FilePath(path) => pcm_decode(&path)?,
            AudioInput::Samples { data, sample_rate } => (data, sample_rate),
        };

        let audio_sample_rate = sample_rate.unwrap_or(audio_sample_rate);

        #[cfg(feature = "onnx")]
        {
            let frontend = self.frontend.as_ref().ok_or_else(|| {
                CosyVoice3Error::FeatureNotAvailable(
                    "ONNX frontend models not found. Please ensure speech_tokenizer_v3.onnx and campplus.onnx are in the model directory.".to_string(),
                )
            })?;

            // Create tensor from audio data
            let audio_tensor =
                Tensor::from_vec(audio_data.clone(), audio_data.len(), &Device::Cpu)
                    .map_err(wrap_candle_err)?;

            // Extract features using frontend
            let (tokens, mel, embedding) = frontend
                .extract_prompt_features(&audio_tensor, audio_sample_rate as usize)
                .map_err(wrap_candle_err)?;

            // Convert tokens to Vec<u32>
            let prompt_tokens: Vec<u32> = tokens
                .flatten_all()
                .map_err(wrap_candle_err)?
                .to_vec1::<i64>()
                .map_err(wrap_candle_err)?
                .into_iter()
                .map(|x| x as u32)
                .collect();

            // Convert mel to Vec<Vec<f32>>
            let mel = mel.to_dtype(DType::F32).map_err(wrap_candle_err)?;
            let mel_shape = mel.dims();
            let t_dim = if mel_shape.len() == 3 {
                mel_shape[1]
            } else {
                mel_shape[0]
            };
            let mel_dim = if mel_shape.len() == 3 {
                mel_shape[2]
            } else {
                mel_shape[1]
            };

            let mel_flat: Vec<f32> = mel
                .flatten_all()
                .map_err(wrap_candle_err)?
                .to_vec1()
                .map_err(wrap_candle_err)?;

            let prompt_mel: Vec<Vec<f32>> = mel_flat
                .chunks(mel_dim)
                .take(t_dim)
                .map(|c| c.to_vec())
                .collect();

            // Convert speaker embedding to Vec<f32>
            let speaker_embedding: Vec<f32> = embedding
                .flatten_all()
                .map_err(wrap_candle_err)?
                .to_dtype(DType::F32)
                .map_err(wrap_candle_err)?
                .to_vec1()
                .map_err(wrap_candle_err)?;

            Ok((prompt_tokens, prompt_mel, speaker_embedding))
        }

        #[cfg(not(feature = "onnx"))]
        {
            let _ = (audio_data, audio_sample_rate);
            Err(CosyVoice3Error::FeatureNotAvailable(
                "Feature extraction requires the 'onnx' feature. Please compile with --features onnx, or use load_prompt_features() with pre-extracted features.".to_string(),
            ).into())
        }
    }

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
