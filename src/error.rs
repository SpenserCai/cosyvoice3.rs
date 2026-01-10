//! Error types for CosyVoice3

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use thiserror::Error;

/// CosyVoice3 error type
#[derive(Error, Debug)]
pub enum CosyVoice3Error {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),
}

impl From<CosyVoice3Error> for PyErr {
    fn from(err: CosyVoice3Error) -> Self {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Wrap a candle error into a PyErr
pub fn wrap_candle_err(err: candle_core::Error) -> PyErr {
    PyRuntimeError::new_err(format!("Candle error: {}", err))
}
