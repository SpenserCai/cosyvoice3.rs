# cosyvoice3

High-performance Python bindings for [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) TTS, powered by Rust and [Candle](https://github.com/huggingface/candle).

## Installation

```bash
pip install cosyvoice3
```

## Quick Start

```python
from cosyvoice3 import CosyVoice3
import wave, struct

# Load model
model = CosyVoice3("./CosyVoice3-0.5B-Candle")

# Zero-shot voice cloning
audio = model.inference_zero_shot(
    text="你好，这是一个测试。",
    prompt_text="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    prompt_wav="prompt.wav"
)

# Save audio
with wave.open("output.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(model.sample_rate)
    audio_int16 = [int(max(-32768, min(32767, s * 32767))) for s in audio]
    f.writeframes(struct.pack(f"{len(audio_int16)}h", *audio_int16))
```

## Model Weights

Download pre-converted weights from Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli download spensercai/CosyVoice3-0.5B-Candle --local-dir ./CosyVoice3-0.5B-Candle
```

## Usage

### Zero-Shot Voice Cloning

```python
audio = model.inference_zero_shot(
    text="Hello, this is synthesized speech.",
    prompt_text="You are a helpful assistant.<|endofprompt|>Hello, this is my voice.",
    prompt_wav="reference.wav"
)
```

### Cross-Lingual Synthesis

```python
audio = model.inference_cross_lingual(
    text="<|en|>Hello, this is cross-lingual synthesis.",
    prompt_wav="chinese_reference.wav"
)
```

### Instruction-Guided Synthesis

```python
audio = model.inference_instruct(
    text="你好世界",
    instruct_text="You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
    prompt_wav="reference.wav"
)
```

### Device Selection

```python
from cosyvoice3 import CosyVoice3, PyDevice

# Auto-select best device
model = CosyVoice3("./model")

# Explicit device selection
model = CosyVoice3("./model", device=PyDevice("cpu"))
model = CosyVoice3("./model", device=PyDevice("cuda"))   # CUDA GPU
model = CosyVoice3("./model", device=PyDevice("metal"))  # macOS Metal

# Check availability
print(f"CUDA: {PyDevice.cuda_is_available()}")
print(f"Metal: {PyDevice.metal_is_available()}")
```

### Custom Sampling

```python
from cosyvoice3 import SamplingConfig

config = SamplingConfig(top_k=25, top_p=0.8, temperature=1.0)
audio = model.inference_zero_shot(
    text="Hello!",
    prompt_text="You are a helpful assistant.<|endofprompt|>Hi.",
    prompt_wav="prompt.wav",
    sampling_config=config,
    n_timesteps=10
)
```

### Text Normalization

```python
from cosyvoice3 import TextNormalizer

# Download FST files: huggingface-cli download mio/wetext --local-dir ./wetext
normalizer = TextNormalizer("./wetext")
text = normalizer.normalize("2024年1月15日，价格是$100.50")
# → "二零二四年一月十五日，价格是一百美元五十美分"
```

## API Reference

| Class | Description |
|-------|-------------|
| `CosyVoice3` | Main TTS model |
| `PyDevice` | Device selection (cpu/cuda/metal) |
| `SamplingConfig` | LLM sampling parameters |
| `TextNormalizer` | Text preprocessing |
| `SynthesisMode` | Synthesis mode enum |

## License

MIT OR Apache-2.0
