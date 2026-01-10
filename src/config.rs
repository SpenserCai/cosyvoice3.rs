//! Configuration types for CosyVoice3

use pyo3::prelude::*;
use serde::Deserialize;
use std::path::Path;

/// Runtime configuration for CosyVoice3
#[pyclass]
#[derive(Clone, Debug, Deserialize)]
pub struct CosyVoice3Config {
    #[pyo3(get)]
    pub sample_rate: usize,
    #[pyo3(get)]
    pub llm_input_size: usize,
    #[pyo3(get)]
    pub llm_output_size: usize,
    #[pyo3(get)]
    pub speech_token_size: usize,
    #[pyo3(get)]
    pub spk_embed_dim: usize,
    #[pyo3(get)]
    pub token_frame_rate: usize,
    #[pyo3(get)]
    pub token_mel_ratio: usize,
    #[pyo3(get)]
    pub chunk_size: usize,
    #[pyo3(get)]
    pub pre_lookahead_len: usize,
    pub dit: DiTConfig,
    pub hift: HiFTConfig,
    pub qwen2: Qwen2Config,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DiTConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub ff_mult: usize,
    pub mel_dim: usize,
    pub spk_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HiFTConfig {
    pub in_channels: usize,
    pub base_channels: usize,
    pub nb_harmonics: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub istft_n_fft: usize,
    pub istft_hop_len: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub source_resblock_kernel_sizes: Vec<usize>,
    pub source_resblock_dilation_sizes: Vec<Vec<usize>>,
    pub conv_pre_look_right: usize,
    pub nsf_alpha: f64,
    pub nsf_sigma: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

#[pymethods]
impl CosyVoice3Config {
    /// Load configuration from a JSON file
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open config file: {}", e))
        })?;
        let config: CosyVoice3Config = serde_json::from_reader(file).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to parse config: {}", e))
        })?;
        Ok(config)
    }

    fn __repr__(&self) -> String {
        format!(
            "CosyVoice3Config(sample_rate={}, speech_token_size={})",
            self.sample_rate, self.speech_token_size
        )
    }
}

impl CosyVoice3Config {
    /// Load from model directory
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, crate::error::CosyVoice3Error> {
        let config_path = model_dir.join("config.json");
        let file = std::fs::File::open(&config_path)?;
        let config: CosyVoice3Config = serde_json::from_reader(file)?;
        Ok(config)
    }

    /// Convert to LLM config
    pub fn to_llm_config(&self) -> candle_transformers::models::cosyvoice::CosyVoice3LMConfig {
        candle_transformers::models::cosyvoice::CosyVoice3LMConfig {
            llm_input_size: self.llm_input_size,
            llm_output_size: self.llm_output_size,
            speech_token_size: self.speech_token_size,
            mix_ratio: (5, 15),
            qwen2: candle_transformers::models::cosyvoice::Qwen2Config {
                hidden_size: self.qwen2.hidden_size,
                num_hidden_layers: self.qwen2.num_hidden_layers,
                num_attention_heads: self.qwen2.num_attention_heads,
                num_key_value_heads: self.qwen2.num_key_value_heads,
                intermediate_size: self.qwen2.intermediate_size,
                max_position_embeddings: 32768,
                rope_theta: self.qwen2.rope_theta,
                rms_norm_eps: self.qwen2.rms_norm_eps,
                vocab_size: self.qwen2.vocab_size,
                tie_word_embeddings: true,
            },
        }
    }

    /// Convert to HiFT config
    pub fn to_hift_config(&self) -> candle_transformers::models::cosyvoice::HiFTConfig {
        candle_transformers::models::cosyvoice::HiFTConfig {
            in_channels: self.hift.in_channels,
            base_channels: self.hift.base_channels,
            nb_harmonics: self.hift.nb_harmonics,
            sampling_rate: self.sample_rate,
            nsf_alpha: self.hift.nsf_alpha,
            nsf_sigma: self.hift.nsf_sigma,
            upsample_rates: self.hift.upsample_rates.clone(),
            upsample_kernel_sizes: self.hift.upsample_kernel_sizes.clone(),
            istft_n_fft: self.hift.istft_n_fft,
            istft_hop_len: self.hift.istft_hop_len,
            resblock_kernel_sizes: self.hift.resblock_kernel_sizes.clone(),
            resblock_dilation_sizes: self.hift.resblock_dilation_sizes.clone(),
            source_resblock_kernel_sizes: self.hift.source_resblock_kernel_sizes.clone(),
            source_resblock_dilation_sizes: self.hift.source_resblock_dilation_sizes.clone(),
            conv_pre_look_right: self.hift.conv_pre_look_right,
        }
    }

    /// Convert to Flow config
    pub fn to_flow_config(&self) -> candle_transformers::models::cosyvoice::FlowConfig {
        candle_transformers::models::cosyvoice::FlowConfig {
            input_size: self.dit.mel_dim,
            output_size: self.dit.mel_dim,
            vocab_size: self.speech_token_size,
            token_mel_ratio: self.token_mel_ratio,
            pre_lookahead_len: self.pre_lookahead_len,
            dit: candle_transformers::models::cosyvoice::DiTConfig {
                dim: self.dit.dim,
                depth: self.dit.depth,
                heads: self.dit.heads,
                dim_head: self.dit.dim_head,
                ff_mult: self.dit.ff_mult,
                mel_dim: self.dit.mel_dim,
                spk_dim: self.dit.spk_dim,
                static_chunk_size: self.chunk_size * self.token_mel_ratio,
            },
            cfm: candle_transformers::models::cosyvoice::CFMConfig::default(),
        }
    }
}
