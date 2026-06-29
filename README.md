# audiopy_ai 🎧

[![PyPI Version](https://img.shields.io/pypi/v/audiopy_ai.svg)](https://pypi.org/project/audiopy_ai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/audiopy_ai.svg)](https://pypi.org/project/audiopy_ai/)
[![License](https://img.shields.io/pypi/l/audiopy_ai.svg)](https://pypi.org/project/audiopy_ai/)

**Open-Source Framework for Intelligent Audio Editing (DSP + AI) — inspired by OpenCV, but designed for audio.**

`audiopy_ai` is a unified Python library that seamlessly bridges the gap between traditional Digital Signal Processing (DSP) and modern deep-learning-based AI analysis. It enables developers to load, manipulate, filter, stretch, and analyze audio files via a clean, high-performance, and consistent API.

---

## ✨ Features

- 📂 **High-Performance I/O**: Fast loading and saving of audio files (WAV, MP3, FLAC, etc.) into unified numpy arrays.
- 🎚️ **DSP Audio Editing**: Trimming, fading, peak-normalization, gain adjustment, reversing, and sample concatenation.
- 🔊 **Advanced DSP Effects**:
  - Butterworth low-pass and high-pass filters (via Scipy SOS filters).
  - Convolution-based reverb with impulse response normalization.
  - Resampling-based time-stretching.
- 🤖 **AI-Powered Analysis**:
  - Speech-to-Text transcription (using Whisper).
  - Audio classification and sound event tagging (using HuBERT / AudioSet models).
- 🔌 **Zero-Config FFmpeg**: Automated local path auto-discovery for FFmpeg (supporting Scoop, Winget, and custom Program Files locations) to ensure seamless audio decoding out-of-the-box on Windows and Unix systems.

---

## 📦 Installation

### 1. Core Install (DSP & I/O)

```bash
pip install audiopy_ai
```

### 2. Advanced Install (With Librosa for Feature Extraction)

```bash
pip install audiopy_ai[advanced]
```

### ⚡ System Requirements

- **Python**: `>= 3.9`
- **FFmpeg**: Required for decoding compressed formats (MP3, M4A, etc.) and video analysis. `audiopy_ai` automatically searches standard environment directories, Winget packages, and Scoop shims to configure FFmpeg on import.

---

## 🚀 Quick Start

Here is a simple example showing how to load a file, normalize it, apply a fade, and run AI-based classification on it:

```python
import audiopy_ai as ap

# 1. Load audio file (WAV, MP3, etc.)
samples, sr = ap.load("input.wav")
print(f"Loaded {len(samples)} samples at {sr}Hz")

# 2. DSP Processing: Normalize amplitude & apply fade
normalized = ap.normalize(samples, target_db=-15.0)
processed = ap.fade(normalized, sr, fade_in_s=0.5, fade_out_s=0.5)

# 3. Save processed file
ap.save("output_processed.wav", processed, sr)
print("Saved processed audio!")

# 4. Run AI Event Analysis
extractor = ap.SoundComponentExtractor(analysis_type="audio_tagging")
analysis = extractor.extract_components("output_processed.wav")

print("Detected Components:")
for item in analysis.get("results", []):
    print(f" • {item['label']}: {item['score']:.1%}")
```

---

## 📖 API Reference

### 🔊 I/O Module

#### `ap.load(path: str, mono: bool = True) -> (np.ndarray, int)`
Loads an audio file into a 1D float32 numpy array. If `mono=True` and the file is stereo, channels are automatically averaged.

```python
samples, sr = ap.load("sound.mp3")
```

#### `ap.save(path: str, samples: np.ndarray, sr: int) -> None`
Saves a numpy array back into a sound file.

```python
ap.save("output.wav", samples, sr)
```

---

### ⚙️ DSP Editing Module

#### `ap.trim(samples: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray`
Trims audio to a specified time window (in seconds).

```python
trimmed = ap.trim(samples, sr, start_s=1.0, end_s=5.0)
```

#### `ap.concat(list_of_samples: list[np.ndarray]) -> np.ndarray`
Concatenates multiple audio sample arrays sequentially.

```python
full_audio = ap.concat([clip1, clip2])
```

#### `ap.apply_gain(samples: np.ndarray, db: float) -> np.ndarray`
Applies a gain adjustment (in decibels) to the audio samples.

```python
louder = ap.apply_gain(samples, db=6.0)
```

#### `ap.normalize(samples: np.ndarray, target_db: float = -20.0) -> np.ndarray`
Normalizes the audio samples to match a target RMS decibel level.

```python
normalized = ap.normalize(samples, target_db=-18.0)
```

#### `ap.fade(samples: np.ndarray, sr: int, fade_in_s: float = 0.5, fade_out_s: float = 0.5) -> np.ndarray`
Applies linear fade-in and fade-out envelopes to the audio.

```python
faded = ap.fade(samples, sr, fade_in_s=1.0, fade_out_s=1.0)
```

#### `ap.reverse(samples: np.ndarray) -> np.ndarray`
Reverses the audio array (plays backwards).

```python
reversed_audio = ap.reverse(samples)
```

#### `ap.rms_db(samples: np.ndarray, eps: float = 1e-9) -> float`
Calculates the Root-Mean-Square (RMS) volume of the audio in decibels.

```python
volume_db = ap.rms_db(samples)
```

---

### 🎚️ Effects Submodule

#### Filters (`audiopy_ai.effects.filters`)

- `butter_lowpass(sr, cutoff_hz, order=4)`: Generates low-pass filter coefficients.
- `butter_highpass(sr, cutoff_hz, order=4)`: Generates high-pass filter coefficients.
- `apply_sos_filter(samples, sos)`: Applies the generated filter coefficients (SOS format) to the audio.

```python
from audiopy_ai.effects.filters import butter_lowpass, apply_sos_filter

# Filter high frequencies out above 1000Hz
sos = butter_lowpass(sr, cutoff_hz=1000)
filtered = apply_sos_filter(samples, sos)
```

#### Reverb (`audiopy_ai.effects.reverb`)

- `reverb(samples, ir, wet=0.3)`: Applies a convolution-based reverb effect using an Impulse Response (IR) audio array.

```python
from audiopy_ai.effects.reverb import reverb

# ir_samples is another audio array loaded from a reverb room impulse response file
wet_audio = reverb(samples, ir_samples, wet=0.4)
```

#### Time stretching (`audiopy_ai.effects.time`)

- `time_stretch(samples, rate)`: Changes the playback speed of the audio without pitch shift (using resampling). `rate > 1.0` speeds it up, while `rate < 1.0` slows it down.

```python
from audiopy_ai.effects.time import time_stretch

fast_audio = time_stretch(samples, rate=1.5)
```

---

### 🤖 AI-Powered Analysis Module

#### `ap.AudioAnalyzer(model_name: str = "facebook/wav2vec2-base", api_key: str = None)`
Initializes the HuggingFace-backed pipeline analyzer.
- **Speech-to-Text models**: `openai/whisper-base`, `facebook/wav2vec2-base`
- **Audio Tagging models**: `superb/hubert-base-superb-ks` (keyword spotting)

```python
from audiopy_ai.ai_analysis import AudioAnalyzer

analyzer = AudioAnalyzer(model_name="openai/whisper-base")
results = analyzer.analyze_audio("speech.wav")
print("Transcribed Text:", results["components"]["metadata"]["transcription"])
```

#### `ap.SoundComponentExtractor(analysis_type: str = "audio_tagging", api_key: str = None)`
Simplified helper wrapper for executing AI analyses.

```python
from audiopy_ai.ai_analysis import SoundComponentExtractor

# Instantiate for Sound Event detection
extractor = SoundComponentExtractor(analysis_type="audio_tagging")
info = extractor.extract_components("recording.wav")
```

---

## 🤝 Contributing

Contributions are always welcome! 

1. Fork the repository on GitHub.
2. Install the library locally in editable mode:
   ```bash
   pip install -e .
   ```
3. Run the unit tests to ensure everything is working:
   ```bash
   pytest tests/
   ```
4. Open a Pull Request with your feature branch.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.