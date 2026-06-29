# audiopy_ai AI Analysis - Quick Reference Guide

## Installation

```bash
pip install transformers torch
```

Or install the full audiopy_ai package with AI capabilities:
```bash
pip install -e .
```

## Quick Start

### 1. Basic Sound Component Detection

```python
from audiopy_ai.ai_analysis import SoundComponentExtractor

# Initialize analyzer
analyzer = SoundComponentExtractor(analysis_type="audio_tagging")

# Analyze audio file
results = analyzer.extract_components("your_audio.wav")

# Get detected sounds
sounds = results['components']['detected_sounds']
scores = results['components']['confidence_scores']

for sound, score in zip(sounds, scores):
    print(f"{sound}: {score:.1%}")
```

### 2. Extract with Audio Features

```python
from audiopy_ai.ai_analysis import SoundComponentExtractor
from audiopy_ai.io import load

# Load audio
samples, sr = load("audio.wav")

# Analyze with features
analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
results = analyzer.extract_components_with_features(
    audio_path="audio.wav",
    samples=samples,
    sr=sr
)

# Access AI results
components = results['components']['detected_sounds']

# Access audio features
features = results['audio_features']
print(f"Spectral Centroid: {features['spectral_centroid']}")
print(f"MFCC Mean: {features['mfcc_mean']}")
```

### 3. Speech Recognition

```python
from audiopy_ai.ai_analysis import SoundComponentExtractor

analyzer = SoundComponentExtractor(analysis_type="speech_recognition")
results = analyzer.extract_components("speech.wav")

transcription = results['components']['metadata']['transcription']
print(f"Transcription: {transcription}")
```

## Available Analysis Types

| Type | Description | Use Case |
|------|-------------|----------|
| `audio_tagging` | Detect sound events in audio | Identify dog barks, music, speech, etc. |
| `speech_recognition` | Convert speech to text | Transcribe spoken content |
| `emotion_detection` | Detect emotions from speech | Analyze speaker sentiment |
| `music_analysis` | Analyze music characteristics | Extract music features |

## Using Custom Models

```python
from audiopy_ai.ai_analysis import AudioAnalyzer

# Use any HuggingFace model
analyzer = AudioAnalyzer(model_name="facebook/wav2vec2-base")
results = analyzer.analyze_audio("audio.wav")
```

## Private Models with API Key

```python
from audiopy_ai.ai_analysis import SoundComponentExtractor

analyzer = SoundComponentExtractor(
    analysis_type="audio_tagging",
    api_key="hf_your_api_key_here"
)

results = analyzer.extract_components("audio.wav")
```

## Batch Processing

```python
from audiopy_ai.ai_analysis import SoundComponentExtractor

analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

results = {}
for audio_file in audio_files:
    try:
        result = analyzer.extract_components(audio_file)
        results[audio_file] = result['components']['detected_sounds']
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
```

## Integration with Audio Editing

```python
from audiopy_ai.io import load, save
from audiopy_ai.ai_analysis import SoundComponentExtractor
from audiopy_ai.edit import normalize, fade

# Load audio
samples, sr = load("audio.wav")

# Analyze
analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
results = analyzer.extract_components_with_features("audio.wav", samples, sr)

# Get detected sounds
sounds = results['components']['detected_sounds']
print(f"Detected: {sounds}")

# Edit based on analysis
processed = normalize(samples)
processed = fade(processed, sr, fade_in_s=1.0)

# Save
save("output.wav", processed, sr)
```

## Available Audio Features

When using `extract_components_with_features()`, you get:

```
audio_features:
  - spectral_centroid: Brightness of sound
  - spectral_rolloff: High-frequency cutoff
  - zero_crossing_rate: Noisiness indicator
  - mfcc_mean: Mel-frequency cepstral coefficients (mean)
  - mfcc_std: MFCC standard deviation
  - onset_strength: Rhythm/beat strength
  - chroma_mean: Musical pitch distribution
```

## Result Structure

```python
results = {
    'model': 'model_name',
    'task': 'audio-classification',
    'components': {
        'detected_sounds': ['dog', 'bark', 'outdoors'],
        'confidence_scores': [0.95, 0.87, 0.72],
        'metadata': {...}
    },
    'audio_features': {
        'spectral_centroid': 3245.67,
        'mfcc_mean': [...],
        ...
    }
}
```

## Performance Notes

- **First Run**: Models will download (300MB-700MB depending on model)
- **GPU**: Install `torch` with CUDA support for faster inference
- **Memory**: Requires ~4GB RAM for most models
- **Speed**: Processing typically takes a few seconds per audio file

## Troubleshooting

### Out of Memory
```python
# Use a smaller model
analyzer = AudioAnalyzer(model_name="facebook/wav2vec2-base")
```

### Slow Processing
- Ensure GPU support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Use smaller model variants
- Process audio in chunks if very long

### Model Not Found
- Check internet connection (model needs to download)
- Verify API key if using private models
- Check HuggingFace model repository

## More Information

For more examples, see: `examples/ai_analysis_examples.py`
For detailed API docs, see docstrings in: `audiopy_ai/ai_analysis.py`
