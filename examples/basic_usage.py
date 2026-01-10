#!/usr/bin/env python3
"""
Basic usage example for cosyvoice3 Python bindings.

This example demonstrates how to use the CosyVoice3 TTS model.
"""

import struct
import wave
from pathlib import Path

# Import the cosyvoice3 module
from cosyvoice3 import (
    CosyVoice3,
    CosyVoice3Config,
    PyDevice,
    SamplingConfig,
    SynthesisMode,
    HAS_CUDA,
    HAS_METAL,
    HAS_ONNX,
)


def main():
    print("CosyVoice3 Python Bindings Example")
    print("=" * 40)
    
    # Print feature flags
    print(f"\nFeature flags:")
    print(f"  HAS_CUDA: {HAS_CUDA}")
    print(f"  HAS_METAL: {HAS_METAL}")
    print(f"  HAS_ONNX: {HAS_ONNX}")
    
    # Device selection
    print(f"\nDevice availability:")
    print(f"  CUDA available: {PyDevice.cuda_is_available()}")
    print(f"  Metal available: {PyDevice.metal_is_available()}")
    print(f"  Best available: {PyDevice.best_available()}")
    
    # Create device
    device = PyDevice.best_available()
    print(f"\nUsing device: {device}")
    
    # Create sampling config
    sampling_config = SamplingConfig(
        top_k=25,
        top_p=0.8,
        temperature=1.0,
        repetition_penalty=1.0,
    )
    print(f"\nSampling config: {sampling_config}")
    
    # Create synthesis mode
    mode = SynthesisMode("zero_shot")
    print(f"Synthesis mode: {mode}")
    
    # Example: Load model (uncomment when you have model weights)
    # model_dir = "/path/to/CosyVoice3-0.5B-Candle"
    # 
    # print(f"\nLoading model from: {model_dir}")
    # model = CosyVoice3(model_dir, device=device, use_f16=False)
    # print(f"Model loaded: {model}")
    # print(f"Sample rate: {model.sample_rate}")
    # print(f"Has ONNX: {model.has_onnx}")
    # 
    # # Example: Synthesize speech
    # # You need to provide prompt features (from pre-extracted safetensors or ONNX extraction)
    # prompt_speech_tokens = [...]  # List of speech tokens
    # prompt_mel = [[...]]  # 2D list: [T, 80] mel spectrogram
    # speaker_embedding = [...]  # List of 192 floats
    # 
    # audio = model.synthesize(
    #     text="Hello, this is a test.",
    #     prompt_speech_tokens=prompt_speech_tokens,
    #     prompt_mel=prompt_mel,
    #     speaker_embedding=speaker_embedding,
    #     prompt_text="You are a helpful assistant.<|endofprompt|>Hello world.",
    #     mode=SynthesisMode.ZeroShot,
    #     sampling_config=sampling_config,
    #     n_timesteps=10,
    # )
    # 
    # # Save audio to WAV file
    # save_wav("output.wav", audio, model.sample_rate)
    # print(f"\nAudio saved to output.wav")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")


def save_wav(filename: str, audio: list, sample_rate: int):
    """Save audio samples to a WAV file."""
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Convert float [-1, 1] to int16
        audio_int16 = [int(max(-32768, min(32767, s * 32767))) for s in audio]
        wav_file.writeframes(struct.pack(f"{len(audio_int16)}h", *audio_int16))


if __name__ == "__main__":
    main()
