"""
CosyVoice3 - Python bindings for CosyVoice3 TTS using Candle

This module provides Python bindings for the CosyVoice3 text-to-speech model,
implemented in Rust using the Candle deep learning framework.

Example:
    >>> from cosyvoice3 import CosyVoice3, SynthesisMode, SamplingConfig, PyDevice
    >>> 
    >>> # Load model
    >>> model = CosyVoice3("path/to/model", device=PyDevice("cpu"))
    >>> 
    >>> # Synthesize speech
    >>> audio = model.synthesize(
    ...     text="Hello, world!",
    ...     prompt_speech_tokens=[...],
    ...     prompt_mel=[[...]],
    ...     speaker_embedding=[...],
    ... )
"""

# Import from the compiled Rust extension
# The actual module is built by maturin and named 'cosyvoice3'
try:
    from cosyvoice3.cosyvoice3 import (
        CosyVoice3,
        CosyVoice3Config,
        PyDevice,
        SynthesisMode,
        SamplingConfig,
        TextNormalizer,
        normalize_text,
        __version__,
        HAS_CUDA,
        HAS_METAL,
        HAS_ONNX,
    )
except ImportError:
    # Fallback for development - try direct import
    from .cosyvoice3 import (  # type: ignore
        CosyVoice3,
        CosyVoice3Config,
        PyDevice,
        SynthesisMode,
        SamplingConfig,
        TextNormalizer,
        normalize_text,
        __version__,
        HAS_CUDA,
        HAS_METAL,
        HAS_ONNX,
    )

__all__ = [
    "CosyVoice3",
    "CosyVoice3Config",
    "PyDevice",
    "SynthesisMode",
    "SamplingConfig",
    "TextNormalizer",
    "normalize_text",
    "__version__",
    "HAS_CUDA",
    "HAS_METAL",
    "HAS_ONNX",
]
