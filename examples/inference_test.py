import wave
import struct
import time
from cosyvoice3 import CosyVoice3, PyDevice, SamplingConfig, TextNormalizer

# Paths
model_dir = './weights/CosyVoice3-0.5B-Candle'
prompt_wav = './prompt_wav/zero_shot_prompt.wav'
output_dir = './outs'
wetext_dir = './weights/wetext'
use_device = PyDevice.best_available()

def save_audio(audio, sample_rate, output_path):
    """Save audio samples to WAV file."""
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        audio_int16 = [int(max(-32768, min(32767, s * 32767))) for s in audio]
        wav_file.writeframes(struct.pack(f'{len(audio_int16)}h', *audio_int16))


def test_zero_shot(model, normalizer):
    """Test zero-shot inference with text normalization."""
    print('\n=== Zero-Shot Test ===')
    
    # Text normalization
    print('Normalizing text...')
    text = '2026.1.12，今天是阳光明媚的一天'
    normalized_text = normalizer.normalize(text)
    print(f'Original: {text}')
    print(f'Normalized: {normalized_text}')
    
    # Inference
    print('Running inference...')
    start = time.time()
    audio = model.inference_zero_shot(
        text=normalized_text,
        prompt_text='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        prompt_wav=prompt_wav,
        n_timesteps=10
    )
    inference_time = time.time() - start
    
    # Stats
    audio_duration = len(audio) / model.sample_rate
    rtf = inference_time / audio_duration
    print(f'Inference completed in {inference_time:.2f}s')
    print(f'Audio samples: {len(audio)}, duration: {audio_duration:.2f}s, RTF: {rtf:.2f}')
    
    # Save
    output_path = f'{output_dir}/rust_pyo3_test.wav'
    save_audio(audio, model.sample_rate, output_path)
    print(f'Saved to {output_path}')


def test_cross_lingual(model):
    """Test cross-lingual inference."""
    print('\n=== Cross-Lingual Test ===')
    
    start = time.time()
    audio = model.inference_cross_lingual(
        text='<|en|>Hello, this is a cross-lingual synthesis test.',
        prompt_wav=prompt_wav,
        n_timesteps=10
    )
    print(f'Cross-lingual inference: {time.time() - start:.2f}s, {len(audio)} samples')
    
    # Save
    output_path = f'{output_dir}/rust_pyo3_cross_lingual.wav'
    save_audio(audio, model.sample_rate, output_path)
    print(f'Saved to {output_path}')


def test_instruct(model):
    """Test instruct inference."""
    print('\n=== Instruct Test ===')
    
    start = time.time()
    audio = model.inference_instruct(
        text='你好世界',
        instruct_text='You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        prompt_wav=prompt_wav,
        n_timesteps=10
    )
    print(f'Instruct inference: {time.time() - start:.2f}s, {len(audio)} samples')
    
    # Save
    output_path = f'{output_dir}/rust_pyo3_instruct.wav'
    save_audio(audio, model.sample_rate, output_path)
    print(f'Saved to {output_path}')


def main():
    print('=== CosyVoice3 PyO3 Test ===')
    print(f'Model: {model_dir}')
    print(f'Prompt: {prompt_wav}')
    
    # Load model
    print('\nLoading model...')
    start = time.time()
    model = CosyVoice3(model_dir, device=use_device, use_f16=False)
    print(f'Model loaded in {time.time() - start:.2f}s')
    print(f'Sample rate: {model.sample_rate}')
    print(f'Has ONNX: {model.has_onnx}')
    
    # Load text normalizer
    normalizer = TextNormalizer(wetext_dir)
    
    # Run all tests
    test_zero_shot(model, normalizer)
    test_cross_lingual(model)
    test_instruct(model)
    
    print('\n=== All Tests Passed! ===')


if __name__ == '__main__':
    main()