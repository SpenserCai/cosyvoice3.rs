//! CosyVoice3 Python bindings using PyO3
//!
//! This crate provides Python bindings for CosyVoice3 TTS model using Candle.

#![allow(clippy::new_without_default)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod config;
mod device;
mod error;
mod model;

use pyo3::prelude::*;

pub use config::CosyVoice3Config;
pub use device::PyDevice;
pub use error::CosyVoice3Error;
pub use model::CosyVoice3;

/// CosyVoice3 Python module
#[pymodule]
fn cosyvoice3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CosyVoice3>()?;
    m.add_class::<CosyVoice3Config>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<SynthesisMode>()?;
    m.add_class::<SamplingConfig>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add feature flags
    m.add("HAS_CUDA", cfg!(feature = "cuda"))?;
    m.add("HAS_METAL", cfg!(feature = "metal"))?;
    m.add("HAS_ONNX", cfg!(feature = "onnx"))?;

    Ok(())
}

/// Synthesis mode for CosyVoice3
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthesisMode {
    /// Zero-shot voice cloning
    ZeroShot,
    /// Cross-lingual voice cloning
    CrossLingual,
    /// Instruction-based synthesis
    Instruct,
}

#[pymethods]
impl SynthesisMode {
    #[new]
    fn new(mode: &str) -> PyResult<Self> {
        match mode.to_lowercase().as_str() {
            "zero_shot" | "zero-shot" | "zeroshot" => Ok(Self::ZeroShot),
            "cross_lingual" | "cross-lingual" | "crosslingual" => Ok(Self::CrossLingual),
            "instruct" => Ok(Self::Instruct),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid synthesis mode: {}. Valid modes: zero_shot, cross_lingual, instruct",
                mode
            ))),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            Self::ZeroShot => "SynthesisMode.ZeroShot".to_string(),
            Self::CrossLingual => "SynthesisMode.CrossLingual".to_string(),
            Self::Instruct => "SynthesisMode.Instruct".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            Self::ZeroShot => "zero_shot".to_string(),
            Self::CrossLingual => "cross_lingual".to_string(),
            Self::Instruct => "instruct".to_string(),
        }
    }
}

/// Sampling configuration for LLM inference
#[pyclass]
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    #[pyo3(get, set)]
    pub top_k: usize,
    #[pyo3(get, set)]
    pub top_p: f32,
    #[pyo3(get, set)]
    pub temperature: f32,
    #[pyo3(get, set)]
    pub repetition_penalty: f32,
}

#[pymethods]
impl SamplingConfig {
    #[new]
    #[pyo3(signature = (top_k=25, top_p=0.8, temperature=1.0, repetition_penalty=1.0))]
    fn new(top_k: usize, top_p: f32, temperature: f32, repetition_penalty: f32) -> Self {
        Self {
            top_k,
            top_p,
            temperature,
            repetition_penalty,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingConfig(top_k={}, top_p={}, temperature={}, repetition_penalty={})",
            self.top_k, self.top_p, self.temperature, self.repetition_penalty
        )
    }
}

impl From<&SamplingConfig> for candle_transformers::models::cosyvoice::SamplingConfig {
    fn from(config: &SamplingConfig) -> Self {
        Self {
            top_k: config.top_k,
            top_p: config.top_p,
            temperature: config.temperature,
            repetition_penalty: config.repetition_penalty,
        }
    }
}
