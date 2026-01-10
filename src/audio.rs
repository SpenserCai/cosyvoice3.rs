//! Audio processing utilities for CosyVoice3

use crate::error::CosyVoice3Error;
use pyo3::prelude::*;

/// Decode audio file to PCM samples
///
/// Returns (samples, sample_rate)
#[cfg(feature = "symphonia")]
pub fn pcm_decode(path: &str) -> Result<(Vec<f32>, u32), CosyVoice3Error> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::conv::FromSample;

    fn conv<T>(
        samples: &mut Vec<f32>,
        data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>,
    ) where
        T: symphonia::core::sample::Sample,
        f32: symphonia::core::conv::FromSample<T>,
    {
        samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
    }

    // Open the media source
    let src = std::fs::File::open(path)?;

    // Create the media source stream
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint
    let hint = symphonia::core::probe::Hint::new();

    // Use default options
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    // Probe the media source
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(|e| CosyVoice3Error::Model(format!("Failed to probe audio: {}", e)))?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| CosyVoice3Error::Model("No supported audio tracks".to_string()))?;

    let dec_opts: DecoderOptions = Default::default();

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|e| CosyVoice3Error::Model(format!("Unsupported codec: {}", e)))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();

    // Decode loop
    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        if packet.track_id() != track_id {
            continue;
        }

        match decoder
            .decode(&packet)
            .map_err(|e| CosyVoice3Error::Model(format!("Decode error: {}", e)))?
        {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }

    Ok((pcm_data, sample_rate))
}

#[cfg(not(feature = "symphonia"))]
pub fn pcm_decode(_path: &str) -> Result<(Vec<f32>, u32), CosyVoice3Error> {
    Err(CosyVoice3Error::FeatureNotAvailable(
        "Audio decoding requires the 'symphonia' feature".to_string(),
    ))
}

/// Load audio from file path or numpy array
pub enum AudioInput {
    /// Path to audio file
    FilePath(String),
    /// Raw PCM samples with sample rate
    Samples { data: Vec<f32>, sample_rate: u32 },
}

impl<'py> FromPyObject<'py> for AudioInput {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Try to extract as string (file path)
        if let Ok(path) = ob.extract::<String>() {
            return Ok(AudioInput::FilePath(path));
        }

        // Try to extract as list of floats
        if let Ok(data) = ob.extract::<Vec<f32>>() {
            // Default sample rate if not provided
            return Ok(AudioInput::Samples {
                data,
                sample_rate: 16000,
            });
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected str (file path) or list of floats (audio samples)",
        ))
    }
}
