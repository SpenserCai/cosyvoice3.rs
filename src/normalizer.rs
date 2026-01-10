//! Text normalization module using wetext-rs

use pyo3::prelude::*;
use std::path::PathBuf;
use wetext_rs::{Language, Normalizer, NormalizerConfig};

/// Parse language string to Language enum
fn parse_language(lang: &str) -> PyResult<Language> {
    match lang.to_lowercase().as_str() {
        "auto" => Ok(Language::Auto),
        "zh" | "chinese" => Ok(Language::Zh),
        "en" | "english" => Ok(Language::En),
        "ja" | "japanese" => Ok(Language::Ja),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid language: {}. Valid options: auto, zh, en, ja",
            lang
        ))),
    }
}

/// Text normalizer for TTS preprocessing
///
/// Converts numbers, dates, currencies, and other non-standard text
/// to spoken form for better TTS quality.
#[pyclass]
pub struct TextNormalizer {
    normalizer: Normalizer,
    fst_dir: PathBuf,
    language: Language,
    remove_erhua: bool,
}

impl TextNormalizer {
    /// Rebuild the normalizer with current settings
    fn rebuild_normalizer(&mut self) {
        let config = NormalizerConfig::new()
            .with_lang(self.language)
            .with_remove_erhua(self.remove_erhua);
        self.normalizer = Normalizer::new(&self.fst_dir, config);
    }
}

#[pymethods]
impl TextNormalizer {
    /// Create a new text normalizer
    ///
    /// Args:
    ///     fst_dir: Path to the directory containing FST files
    ///     lang: Language code ("auto", "zh", "en", "ja"). Default: "auto"
    ///     remove_erhua: Whether to remove erhua (儿化音). Default: False
    #[new]
    #[pyo3(signature = (fst_dir, lang="auto", remove_erhua=false))]
    fn new(fst_dir: &str, lang: &str, remove_erhua: bool) -> PyResult<Self> {
        let language = parse_language(lang)?;
        let fst_path = PathBuf::from(fst_dir);

        let config = NormalizerConfig::new()
            .with_lang(language)
            .with_remove_erhua(remove_erhua);

        let normalizer = Normalizer::new(&fst_path, config);

        Ok(Self {
            normalizer,
            fst_dir: fst_path,
            language,
            remove_erhua,
        })
    }

    /// Normalize text for TTS
    ///
    /// Args:
    ///     text: Input text to normalize
    ///
    /// Returns:
    ///     Normalized text with numbers, dates, etc. converted to spoken form
    ///
    /// Examples:
    ///     - "2024年" → "二零二四年"
    ///     - "$100.50" → "一百美元五十美分"
    ///     - "3.14" → "三点一四"
    fn normalize(&mut self, text: &str) -> PyResult<String> {
        self.normalizer.normalize(text).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Normalization failed: {}", e))
        })
    }

    /// Set the language for normalization
    ///
    /// Args:
    ///     lang: Language code ("auto", "zh", "en", "ja")
    ///
    /// Note: This recreates the normalizer with the new language setting.
    fn set_language(&mut self, lang: &str) -> PyResult<()> {
        self.language = parse_language(lang)?;
        self.rebuild_normalizer();
        Ok(())
    }

    /// Set whether to remove erhua (儿化音)
    ///
    /// Args:
    ///     remove: Whether to remove erhua
    ///
    /// Note: This recreates the normalizer with the new setting.
    fn set_remove_erhua(&mut self, remove: bool) {
        self.remove_erhua = remove;
        self.rebuild_normalizer();
    }

    /// Get the current language setting
    #[getter]
    fn language(&self) -> String {
        match self.language {
            Language::Auto => "auto".to_string(),
            Language::Zh => "zh".to_string(),
            Language::En => "en".to_string(),
            Language::Ja => "ja".to_string(),
        }
    }

    /// Get the current remove_erhua setting
    #[getter]
    fn remove_erhua_enabled(&self) -> bool {
        self.remove_erhua
    }

    fn __repr__(&self) -> String {
        format!(
            "TextNormalizer(fst_dir='{}', lang='{}', remove_erhua={})",
            self.fst_dir.display(),
            self.language(),
            self.remove_erhua
        )
    }
}

/// Convenience function to normalize text
///
/// Args:
///     fst_dir: Path to the directory containing FST files
///     text: Input text to normalize
///     lang: Language code ("auto", "zh", "en", "ja"). Default: "auto"
///     remove_erhua: Whether to remove erhua. Default: False
///
/// Returns:
///     Normalized text
#[pyfunction]
#[pyo3(signature = (fst_dir, text, lang="auto", remove_erhua=false))]
pub fn normalize_text(
    fst_dir: &str,
    text: &str,
    lang: &str,
    remove_erhua: bool,
) -> PyResult<String> {
    let mut normalizer = TextNormalizer::new(fst_dir, lang, remove_erhua)?;
    normalizer.normalize(text)
}
