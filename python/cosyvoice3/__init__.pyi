"""
Type stubs for cosyvoice3 module.

This file provides type hints for Python IDEs and type checkers.
"""

from typing import List, Optional, Tuple

__version__: str
HAS_CUDA: bool
HAS_METAL: bool
HAS_ONNX: bool

class SynthesisMode:
    """Synthesis mode for CosyVoice3."""
    
    ZeroShot: "SynthesisMode"
    """Zero-shot voice cloning mode."""
    
    CrossLingual: "SynthesisMode"
    """Cross-lingual voice cloning mode."""
    
    Instruct: "SynthesisMode"
    """Instruction-based synthesis mode."""
    
    def __init__(self, mode: str) -> None:
        """
        Create a synthesis mode.
        
        Args:
            mode: Mode name. One of "zero_shot", "cross_lingual", "instruct".
        
        Raises:
            ValueError: If mode is invalid.
        """
        ...
    
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class SamplingConfig:
    """Sampling configuration for LLM inference."""
    
    top_k: int
    """Top-k sampling parameter."""
    
    top_p: float
    """Top-p (nucleus) sampling parameter."""
    
    temperature: float
    """Sampling temperature."""
    
    repetition_penalty: float
    """Repetition penalty."""
    
    def __init__(
        self,
        top_k: int = 25,
        top_p: float = 0.8,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> None:
        """
        Create a sampling configuration.
        
        Args:
            top_k: Top-k sampling parameter. Default: 25.
            top_p: Top-p (nucleus) sampling parameter. Default: 0.8.
            temperature: Sampling temperature. Default: 1.0.
            repetition_penalty: Repetition penalty. Default: 1.0.
        """
        ...
    
    def __repr__(self) -> str: ...

class PyDevice:
    """Device type for computation."""
    
    Cpu: "PyDevice"
    """CPU device."""
    
    Cuda: "PyDevice"
    """CUDA GPU device."""
    
    Metal: "PyDevice"
    """Metal GPU device (macOS)."""
    
    def __init__(self, device: str = "cpu") -> None:
        """
        Create a device.
        
        Args:
            device: Device name. One of "cpu", "cuda", "metal".
        
        Raises:
            ValueError: If device is invalid.
            RuntimeError: If requested device is not available.
        """
        ...
    
    @staticmethod
    def cuda_is_available() -> bool:
        """Check if CUDA is available."""
        ...
    
    @staticmethod
    def metal_is_available() -> bool:
        """Check if Metal is available."""
        ...
    
    @staticmethod
    def best_available() -> "PyDevice":
        """Get the best available device (Metal > CUDA > CPU)."""
        ...
    
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class CosyVoice3Config:
    """Runtime configuration for CosyVoice3."""
    
    sample_rate: int
    """Audio sample rate (typically 24000)."""
    
    llm_input_size: int
    """LLM input embedding size."""
    
    llm_output_size: int
    """LLM output embedding size."""
    
    speech_token_size: int
    """Speech token vocabulary size."""
    
    spk_embed_dim: int
    """Speaker embedding dimension."""
    
    token_frame_rate: int
    """Token frame rate."""
    
    token_mel_ratio: int
    """Token to mel ratio."""
    
    chunk_size: int
    """Chunk size for streaming."""
    
    pre_lookahead_len: int
    """Pre-lookahead length."""
    
    @staticmethod
    def from_file(path: str) -> "CosyVoice3Config":
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to the config.json file.
        
        Returns:
            CosyVoice3Config instance.
        
        Raises:
            IOError: If file cannot be read.
            ValueError: If JSON is invalid.
        """
        ...
    
    def __repr__(self) -> str: ...

class CosyVoice3:
    """CosyVoice3 TTS model."""
    
    sample_rate: int
    """Audio sample rate (read-only)."""
    
    config: CosyVoice3Config
    """Model configuration (read-only)."""
    
    has_onnx: bool
    """Whether ONNX feature is compiled (read-only)."""
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[PyDevice] = None,
        use_f16: bool = False,
    ) -> None:
        """
        Load CosyVoice3 model from a directory.
        
        Args:
            model_dir: Path to the model directory containing weights and config.
            device: Device to run the model on. If None, auto-selects best available.
            use_f16: Whether to use FP16 precision (GPU only).
        
        Raises:
            IOError: If model files cannot be loaded.
            RuntimeError: If model initialization fails.
        """
        ...
    
    def synthesize(
        self,
        text: str,
        prompt_speech_tokens: List[int],
        prompt_mel: List[List[float]],
        speaker_embedding: List[float],
        prompt_text: str = "You are a helpful assistant.<|endofprompt|>",
        mode: SynthesisMode = ...,
        instruct_text: Optional[str] = None,
        sampling_config: Optional[SamplingConfig] = None,
        n_timesteps: int = 10,
    ) -> List[float]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize.
            prompt_speech_tokens: Speech tokens from prompt audio.
            prompt_mel: Mel spectrogram from prompt audio, shape [T, 80].
            speaker_embedding: Speaker embedding, length 192.
            prompt_text: Prompt text for zero-shot mode.
            mode: Synthesis mode (ZeroShot, CrossLingual, Instruct).
            instruct_text: Instruction text for instruct mode.
            sampling_config: Sampling configuration for LLM.
            n_timesteps: Number of CFM sampling steps.
        
        Returns:
            Audio waveform as a list of floats (normalized to [-1, 1]).
        
        Raises:
            RuntimeError: If synthesis fails.
            ValueError: If arguments are invalid.
        """
        ...
    
    def __repr__(self) -> str: ...

__all__ = [
    "CosyVoice3",
    "CosyVoice3Config",
    "PyDevice",
    "SynthesisMode",
    "SamplingConfig",
    "__version__",
    "HAS_CUDA",
    "HAS_METAL",
    "HAS_ONNX",
]
