//! Device management for CosyVoice3

use candle_core::Device;
use pyo3::prelude::*;
use std::sync::Mutex;

static CUDA_DEVICE: Mutex<Option<Device>> = Mutex::new(None);
static METAL_DEVICE: Mutex<Option<Device>> = Mutex::new(None);

/// Device type for computation
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyDevice {
    /// CPU device
    Cpu,
    /// CUDA GPU device
    Cuda,
    /// Metal GPU device (macOS)
    Metal,
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (device="cpu"))]
    fn new(device: &str) -> PyResult<Self> {
        match device.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => {
                if cfg!(feature = "cuda") {
                    Ok(Self::Cuda)
                } else {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "CUDA support not compiled. Rebuild with --features cuda",
                    ))
                }
            }
            "metal" | "mps" => {
                if cfg!(feature = "metal") {
                    Ok(Self::Metal)
                } else {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Metal support not compiled. Rebuild with --features metal",
                    ))
                }
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid device: {}. Valid devices: cpu, cuda, metal",
                device
            ))),
        }
    }

    /// Check if CUDA is available
    #[staticmethod]
    fn cuda_is_available() -> bool {
        cfg!(feature = "cuda") && candle_core::utils::cuda_is_available()
    }

    /// Check if Metal is available
    #[staticmethod]
    fn metal_is_available() -> bool {
        cfg!(feature = "metal") && candle_core::utils::metal_is_available()
    }

    /// Get the best available device
    #[staticmethod]
    pub fn best_available() -> Self {
        if cfg!(feature = "metal") && candle_core::utils::metal_is_available() {
            return Self::Metal;
        }

        if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            return Self::Cuda;
        }

        Self::Cpu
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Cpu => "Device.Cpu".to_string(),
            Self::Cuda => "Device.Cuda".to_string(),
            Self::Metal => "Device.Metal".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Cuda => "cuda".to_string(),
            Self::Metal => "metal".to_string(),
        }
    }
}

impl PyDevice {
    /// Convert to candle Device
    pub fn to_candle_device(&self) -> PyResult<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => Self::get_cuda_device(),
            Self::Metal => Self::get_metal_device(),
        }
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device() -> PyResult<Device> {
        let mut device = CUDA_DEVICE.lock().unwrap();
        if let Some(d) = device.as_ref() {
            return Ok(d.clone());
        }
        let d = Device::new_cuda(0).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create CUDA device: {}",
                e
            ))
        })?;
        *device = Some(d.clone());
        Ok(d)
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::let_underscore_lock)]
    fn get_cuda_device() -> PyResult<Device> {
        let _guard = CUDA_DEVICE.lock(); // Reference to suppress unused warning
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not compiled",
        ))
    }

    #[cfg(feature = "metal")]
    fn get_metal_device() -> PyResult<Device> {
        let mut device = METAL_DEVICE.lock().unwrap();
        if let Some(d) = device.as_ref() {
            return Ok(d.clone());
        }
        let d = Device::new_metal(0).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create Metal device: {}",
                e
            ))
        })?;
        *device = Some(d.clone());
        Ok(d)
    }

    #[cfg(not(feature = "metal"))]
    #[allow(clippy::let_underscore_lock)]
    fn get_metal_device() -> PyResult<Device> {
        let _guard = METAL_DEVICE.lock(); // Reference to suppress unused warning
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal support not compiled",
        ))
    }
}
