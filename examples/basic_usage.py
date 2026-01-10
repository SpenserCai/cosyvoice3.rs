#!/usr/bin/env python3
"""
Basic usage example for cosyvoice3 Python bindings.

This example demonstrates how to use the CosyVoice3 TTS model with the simplified API.
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
    
    # =========================================================================
    # Example: Simplified API (recommended)
    # =========================================================================
    # 
    # model_dir = "/path/to/CosyVoice3-0.5B-Candle"
    # prompt_wav = "/path/to/prompt.wav"
    # 
    # print(f"\nLoading model from: {model_dir}")
    # model = CosyVoice3(model_dir, device=device, use_f16=False)
    # print(f"Model loaded: {model}")
    # print(f"Sample rate: {model.sample_rate}")
    # print(f"Has ONNX: {model.has_onnx}")
    # 
    # # Zero-shot voice cloning - simplest API
    # audio = model.inference_zero_shot(
    #     text="Hello, this is a test.",
    #     prompt_text="You are a helpful assistant.<|endofprompt|>Hello world.",
    #     prompt_wav=prompt_wav,
    # )
    # save_wav("zero_shot_output.wav", audio, model.sample_rate)
    # print("Zero-shot synthesis saved to zero_shot_output.wav")
    # 
    # # Cross-lingual voice cloning
    # audio = model.inference_cross_lingual(
    #     text="<|en|>Hello, this is cross-lingual synthesis.",
    #     prompt_wav=prompt_wav,
    # )
    # save_wav("cross_lingual_output.wav", audio, model.sample_rate)
    # print("Cross-lingual synthesis saved to cross_lingual_output.wav")
    # 
    # # Instruction-based synthesis
    # audio = model.inference_instruct(
    #     text="你好世界",
    #     instruct_text="You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
    #     prompt_wav=prompt_wav,
    # )
    # save_wav("instruct_output.wav", audio, model.sample_rate)
    # print("Instruction-based synthesis saved to instruct_output.wav")
    
    # =========================================================================
    # Example: Advanced API with pre-extracted features
    # =========================================================================
    # 
    # # Load pre-extracted features (useful for reusing the same prompt)
    # features_path = "/path/to/features.safetensors"
    # prompt_speech_tokens, prompt_mel, speaker_embedding = model.load_prompt_features(features_path)
    # 
    # # Use the low-level synthesize API
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
    # save_wav("advanced_output.wav", audio, model.sample_rate)
    
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
