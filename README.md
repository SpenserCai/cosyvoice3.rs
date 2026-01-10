# CosyVoice3-Py

Python bindings for [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) TTS using [Candle](https://github.com/huggingface/candle).

## Features

- 🚀 High-performance Rust implementation
- 🐍 Native Python bindings via PyO3
- 🎯 Zero-shot voice cloning
- 🌍 Cross-lingual synthesis
- 📝 Instruction-guided synthesis
- 💻 Multi-platform support (Windows, macOS, Linux)
- 🔧 GPU acceleration (CUDA, Metal)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/SpenserCai/candle
cd candle

# Install with maturin
pip install maturin
cd cosyvoice3.rs

# Build and install (CPU only)
maturin develop --release

# Build with Metal support (macOS)
maturin develop --release --features metal

# Build with CUDA support (Linux/Windows)
maturin develop --release --features cuda
```

### Build Wheel

```bash
# Build wheel for distribution
maturin build --release

# With specific features
maturin build --release --features "metal,onnx"
```

## Usage

### Basic Usage

```python
from cosyvoice3 import CosyVoice3, SynthesisMode, SamplingConfig, PyDevice

# Load model
model = CosyVoice3(
    model_dir="path/to/CosyVoice3-0.5B-Candle",
    device=PyDevice("cpu"),  # or "cuda", "metal"
    use_f16=False
)

# Prepare prompt features (from pre-extracted safetensors or ONNX extraction)
prompt_speech_tokens = [...]  # List of speech tokens
prompt_mel = [[...]]  # 2D list: [T, 80] mel spectrogram
speaker_embedding = [...]  # List of 192 floats

# Synthesize speech
audio = model.synthesize(
    text="Hello, this is a test.",
    prompt_speech_tokens=prompt_speech_tokens,
    prompt_mel=prompt_mel,
    speaker_embedding=speaker_embedding,
    prompt_text="You are a helpful assistant.<|endofprompt|>Hello world.",
    mode=SynthesisMode.ZeroShot,
    n_timesteps=10
)

# Save audio
import wave
import struct

with wave.open("output.wav", "w") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(model.sample_rate)
    
    # Convert float to int16
    audio_int16 = [int(max(-32768, min(32767, s * 32767))) for s in audio]
    wav_file.writeframes(struct.pack(f"{len(audio_int16)}h", *audio_int16))
```

### Extract Prompt Features (requires ONNX)

```python
from cosyvoice3 import CosyVoice3, PyDevice

# Load model with ONNX support
model = CosyVoice3("path/to/model", device=PyDevice("cpu"))

# Check if frontend is available
if model.has_frontend:
    # Load audio (as float samples)
    audio_data = [...]  # List of float samples
    sample_rate = 24000
    
    # Extract features
    tokens, mel, embedding = model.extract_prompt_features(audio_data, sample_rate)
    
    print(f"Tokens: {len(tokens)}")
    print(f"Mel shape: {len(mel)} x {len(mel[0])}")
    print(f"Embedding: {len(embedding)}")
```

### Synthesis Modes

```python
from cosyvoice3 import SynthesisMode

# Zero-shot voice cloning
audio = model.synthesize(
    text="Text to synthesize",
    mode=SynthesisMode.ZeroShot,
    prompt_text="You are a helpful assistant.<|endofprompt|>Transcript of prompt audio.",
    ...
)

# Cross-lingual synthesis
audio = model.synthesize(
    text="跨语言合成",
    mode=SynthesisMode.CrossLingual,
    ...
)

# Instruction-guided synthesis
audio = model.synthesize(
    text="Hello!",
    mode=SynthesisMode.Instruct,
    instruct_text="Speak in a cheerful tone.",
    ...
)
```

### Sampling Configuration

```python
from cosyvoice3 import SamplingConfig

config = SamplingConfig(
    top_k=25,
    top_p=0.8,
    temperature=1.0,
    repetition_penalty=1.0
)

audio = model.synthesize(
    text="Hello",
    sampling_config=config,
    ...
)
```

### Device Selection

```python
from cosyvoice3 import PyDevice

# CPU
device = PyDevice("cpu")

# CUDA (requires cuda feature)
device = PyDevice("cuda")

# Metal (requires metal feature, macOS only)
device = PyDevice("metal")

# Auto-select best available
device = PyDevice.best_available()

# Check availability
print(f"CUDA available: {PyDevice.cuda_is_available()}")
print(f"Metal available: {PyDevice.metal_is_available()}")
```

## API Reference

### CosyVoice3

Main model class.

**Constructor:**
- `model_dir: str` - Path to model directory
- `device: PyDevice = None` - Device to use (auto-selects if None)
- `use_f16: bool = False` - Use FP16 precision (GPU only)

**Methods:**
- `synthesize(...)` - Synthesize speech from text
- `extract_prompt_features(audio_data, sample_rate)` - Extract features from audio (ONNX required)

**Properties:**
- `sample_rate: int` - Audio sample rate (24000)
- `config: CosyVoice3Config` - Model configuration
- `has_frontend: bool` - Whether ONNX frontend is available

### SynthesisMode

Enum for synthesis modes:
- `ZeroShot` - Zero-shot voice cloning
- `CrossLingual` - Cross-lingual synthesis
- `Instruct` - Instruction-guided synthesis

### SamplingConfig

LLM sampling configuration:
- `top_k: int = 25`
- `top_p: float = 0.8`
- `temperature: float = 1.0`
- `repetition_penalty: float = 1.0`

### PyDevice

Device selection:
- `PyDevice("cpu")` - CPU
- `PyDevice("cuda")` - CUDA GPU
- `PyDevice("metal")` - Metal GPU (macOS)
- `PyDevice.best_available()` - Auto-select best
- `PyDevice.cuda_is_available()` - Check CUDA
- `PyDevice.metal_is_available()` - Check Metal

## Model Weights

Download pre-converted weights from Hugging Face:

```bash
# Using huggingface-cli
huggingface-cli download spensercai/CosyVoice3-0.5B-Candle --local-dir ./model
```

## Building for Different Platforms

### macOS (Metal)

```bash
maturin build --release --features "metal,onnx"
```

### Linux/Windows (CUDA)

```bash
maturin build --release --features "cuda,onnx"
```

### CPU Only

```bash
maturin build --release --features "onnx"
```

## License

MIT OR Apache-2.0

## Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Original model
- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python bindings
