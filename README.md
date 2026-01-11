# cosyvoice3.rs

Python bindings for [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) TTS using [Candle](https://github.com/huggingface/candle).

CosyVoice3 is a state-of-the-art multilingual zero-shot text-to-speech model from FunAudioLLM.

## Features

- � Hirgh-performance Rust implementation via Candle
- 🐍 Native Python bindings via PyO3
- 🎯 Zero-shot voice cloning
- 🌍 Cross-lingual synthesis
- �  Instruction-guided synthesis
- 💻 Multi-platform support (Windows, macOS, Linux)
- 🔧 GPU acceleration (CUDA, Metal)
- 🎵 Built-in audio file loading (WAV, MP3, OGG)

## Model Weights

Pre-converted weights are available on Hugging Face:

**[spensercai/CosyVoice3-0.5B-Candle](https://huggingface.co/spensercai/CosyVoice3-0.5B-Candle)**

```bash
# Download using hf CLI
pip install huggingface_hub
hf download spensercai/CosyVoice3-0.5B-Candle --local-dir ./CosyVoice3-0.5B-Candle
```

### Convert from Original Weights

If you want to convert from the original PyTorch weights:

```bash
# Download original weights
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./Fun-CosyVoice3-0.5B-2512

# Convert to Candle format
python scripts/convert_weights.py \
    --input ./Fun-CosyVoice3-0.5B-2512 \
    --output ./CosyVoice3-0.5B-Candle
```

## Installation

### Pre-built Wheels (Recommended)

Download pre-built wheels from [GitHub Releases](https://github.com/SpenserCai/cosyvoice3.rs/releases):

```bash
# CPU version (all platforms)
pip install cosyvoice3-<version>-cp310-abi3-<platform>.whl

# Metal version (macOS Apple Silicon)
pip install cosyvoice3-<version>-metal-cp310-abi3-macosx_11_0_arm64.whl

# CUDA 11.8+ version (Linux only, maximum compatibility)
pip install cosyvoice3-<version>-cu118-cp310-abi3-<platform>.whl

# CUDA 12.4+ version (Linux/Windows)
pip install cosyvoice3-<version>-cu124-cp310-abi3-<platform>.whl
```

### From Source

```bash
# Clone the repository
git clone https://github.com/SpenserCai/cosyvoice3.rs
cd cosyvoice3.rs

# Install maturin
pip install maturin

# Build and install (default: CPU + ONNX)
maturin develop --release

# Build with Metal support (macOS Apple Silicon)
maturin develop --release --features metal

# Build with CUDA support (Linux/Windows)
maturin develop --release --features cuda
```

#### Linux CUDA Build Notes

**Recommended**: For Linux users with CUDA, building from source is recommended over pre-built wheels for better compatibility.

**Prerequisites**:

1. **Install Rust toolchain**:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. **Install protoc** (Protocol Buffers compiler):
```bash
# Ubuntu/Debian
sudo apt install protobuf-compiler

# Or download from GitHub releases
PB_REL="https://github.com/protocolbuffers/protobuf/releases"
curl -LO $PB_REL/download/v28.3/protoc-28.3-linux-x86_64.zip
sudo unzip protoc-28.3-linux-x86_64.zip -d /usr/local
```

3. **Install maturin**:
```bash
pip install maturin
```

**Build**:
```bash
# Install with editable mode
pip install -e .

# Or use maturin directly
maturin develop --release --features cuda
```

**GCC Version Compatibility**: CUDA has strict GCC version requirements:
- CUDA 11.x: requires GCC ≤ 11
- CUDA 12.x: requires GCC ≤ 12

If your system default GCC is newer (e.g., GCC 13), you need to specify a compatible version:

```bash
# Check your CUDA version
nvcc --version

# Check your GCC version
gcc --version

# If GCC is too new, install a compatible version and set NVCC_CCBIN
# For Ubuntu/Debian:
sudo apt install gcc-11 g++-11  # For CUDA 11.x
sudo apt install gcc-12 g++-12  # For CUDA 12.x

# Build with specific GCC version
NVCC_CCBIN=/usr/bin/gcc-11 maturin develop --release --features cuda  # CUDA 11.x
NVCC_CCBIN=/usr/bin/gcc-12 maturin develop --release --features cuda  # CUDA 12.x
```

**Environment Variables for CUDA Build**:
| Variable | Description | Example |
|----------|-------------|---------|
| `NVCC_CCBIN` | Path to GCC compiler for nvcc | `/usr/bin/gcc-11` |
| `CUDA_COMPUTE_CAP` | Target GPU compute capability | `80` (for A100), `89` (for RTX 4090) |
| `CUDA_PATH` | CUDA installation path (usually auto-detected) | `/usr/local/cuda` |

### Build Wheel

```bash
# Build wheel for distribution
maturin build --release

# With specific features
maturin build --release --features "metal"
```

## Quick Start

```python
from cosyvoice3 import CosyVoice3

# Load model
model = CosyVoice3("./CosyVoice3-0.5B-Candle")

# Zero-shot voice cloning - just provide text and a prompt audio file
audio = model.inference_zero_shot(
    text="你好，这是一个测试。",
    prompt_text="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    prompt_wav="prompt.wav"
)

# Save audio
import wave, struct
with wave.open("output.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(model.sample_rate)
    audio_int16 = [int(max(-32768, min(32767, s * 32767))) for s in audio]
    f.writeframes(struct.pack(f"{len(audio_int16)}h", *audio_int16))
```

## Usage

### Zero-Shot Voice Cloning

Clone a voice from a reference audio sample:

```python
from cosyvoice3 import CosyVoice3, SamplingConfig

model = CosyVoice3("./CosyVoice3-0.5B-Candle")

# Basic usage
audio = model.inference_zero_shot(
    text="Hello, this is synthesized speech.",
    prompt_text="You are a helpful assistant.<|endofprompt|>Hello, this is my voice sample.",
    prompt_wav="reference_voice.wav"
)

# With custom sampling config
config = SamplingConfig(top_k=25, top_p=0.8, temperature=1.0)
audio = model.inference_zero_shot(
    text="Hello, this is synthesized speech.",
    prompt_text="You are a helpful assistant.<|endofprompt|>Hello, this is my voice sample.",
    prompt_wav="reference_voice.wav",
    sampling_config=config,
    n_timesteps=10
)
```

### Cross-Lingual Voice Cloning

Clone a voice across different languages:

```python
audio = model.inference_cross_lingual(
    text="<|en|>Hello, this is cross-lingual synthesis.",
    prompt_wav="chinese_reference.wav"
)
```

### Instruction-Guided Synthesis

Control speech style with instructions:

```python
audio = model.inference_instruct(
    text="你好世界",
    instruct_text="You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
    prompt_wav="reference.wav"
)
```

### Using Pre-extracted Features

For repeated synthesis with the same voice, you can load pre-extracted features:

```python
# Load features from safetensors file
tokens, mel, embedding = model.load_prompt_features("features.safetensors")

# Use low-level synthesize API
from cosyvoice3 import SynthesisMode

audio = model.synthesize(
    text="Hello, world!",
    prompt_speech_tokens=tokens,
    prompt_mel=mel,
    speaker_embedding=embedding,
    prompt_text="You are a helpful assistant.<|endofprompt|>Hello.",
    mode=SynthesisMode.ZeroShot
)
```

### Device Selection

```python
from cosyvoice3 import CosyVoice3, PyDevice

# Auto-select best available device
model = CosyVoice3("./model")

# Explicitly specify device
model = CosyVoice3("./model", device=PyDevice("cpu"))
model = CosyVoice3("./model", device=PyDevice("cuda"))   # Requires cuda feature
model = CosyVoice3("./model", device=PyDevice("metal"))  # Requires metal feature, macOS only

# Use FP16 precision (GPU only)
model = CosyVoice3("./model", device=PyDevice("metal"), use_f16=True)

# Check device availability
print(f"CUDA available: {PyDevice.cuda_is_available()}")
print(f"Metal available: {PyDevice.metal_is_available()}")
print(f"Best available: {PyDevice.best_available()}")
```

### Text Normalization

For better TTS quality, use the built-in text normalizer:

```python
from cosyvoice3 import TextNormalizer

# Initialize normalizer (requires wetext FST files)
normalizer = TextNormalizer("./wetext-fsts")

# Normalize text
text = normalizer.normalize("2024年1月15日，价格是$100.50")
# Output: "二零二四年一月十五日，价格是一百美元五十美分"

# Use normalized text for synthesis
audio = model.inference_zero_shot(
    text=text,
    prompt_text="...",
    prompt_wav="prompt.wav"
)
```

WeText FST files are available on ModelScope:

```bash
git clone https://www.modelscope.cn/models/pengzhendong/wetext
```

## API Reference

### CosyVoice3

Main model class.

**Constructor:**
```python
CosyVoice3(
    model_dir: str,
    device: PyDevice = None,  # Auto-selects if None
    use_f16: bool = False     # Use FP16 precision (GPU only)
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `inference_zero_shot(text, prompt_text, prompt_wav, ...)` | Zero-shot voice cloning |
| `inference_cross_lingual(text, prompt_wav, ...)` | Cross-lingual synthesis |
| `inference_instruct(text, instruct_text, prompt_wav, ...)` | Instruction-guided synthesis |
| `synthesize(text, prompt_speech_tokens, prompt_mel, speaker_embedding, ...)` | Low-level synthesis API |
| `load_prompt_features(features_path)` | Load pre-extracted features |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `sample_rate` | `int` | Audio sample rate (24000) |
| `config` | `CosyVoice3Config` | Model configuration |
| `has_onnx` | `bool` | Whether ONNX feature is compiled |

### SynthesisMode

Enum for synthesis modes:
- `SynthesisMode.ZeroShot` - Zero-shot voice cloning
- `SynthesisMode.CrossLingual` - Cross-lingual synthesis
- `SynthesisMode.Instruct` - Instruction-guided synthesis

### SamplingConfig

LLM sampling configuration:

```python
SamplingConfig(
    top_k: int = 25,
    top_p: float = 0.8,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0
)
```

### PyDevice

Device selection:

```python
PyDevice("cpu")              # CPU
PyDevice("cuda")             # CUDA GPU
PyDevice("metal")            # Metal GPU (macOS)
PyDevice.best_available()    # Auto-select best
PyDevice.cuda_is_available() # Check CUDA
PyDevice.metal_is_available() # Check Metal
```

## About `prompt_text`

In zero-shot mode, `prompt_text` should follow this format:

```
You are a helpful assistant.<|endofprompt|>[transcript of prompt audio]
```

- **Fixed prefix**: `You are a helpful assistant.<|endofprompt|>` - Required by CosyVoice3's LLM
- **Transcript**: The actual text content spoken in the prompt audio

**Example**: If your prompt audio says "希望你以后能够做的比我还好呦", use:

```python
prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
```

For best voice cloning quality, the transcript should accurately match the audio content.

## Build Features

| Feature | Description |
|---------|-------------|
| `onnx` | Enable ONNX frontend for prompt feature extraction (default) |
| `symphonia` | Enable audio file loading (WAV, MP3, OGG) (default) |
| `metal` | Enable Metal GPU acceleration (macOS) |
| `cuda` | Enable CUDA GPU acceleration |
| `accelerate` | Enable Apple Accelerate framework |
| `mkl` | Enable Intel MKL |

## Performance

| Device | RTF (Real-Time Factor) |
|--------|------------------------|
| Apple M1 Pro (Metal) | ~0.3-0.5x |
| CPU (x86_64) | ~2-4x |

*RTF < 1.0 means faster than real-time*

## License

Apache-2.0

## Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Original model by FunAudioLLM
- [Candle](https://github.com/huggingface/candle) - Rust ML framework by Hugging Face
- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python bindings
