# 🎵 audiopy

> **The Python Audio ML Library** — Like OpenCV, but for audio.

`audiopy` is a production-ready Python library for audio loading, processing, feature extraction, effects, visualization, and AI-powered analysis. It provides a simple, unified API that covers the entire audio ML pipeline — from raw file I/O to deep-learning-based transcription, emotion recognition, and source separation.

---

## ✨ Features

| Category | Capabilities |
|---|---|
| **I/O** | Load/save MP3, WAV, FLAC, OGG, M4A; record from mic; download from URL |
| **Core Processing** | Resample, trim, pad, mono/stereo, mix, normalize, split on silence |
| **Feature Extraction** | MFCC, Mel spectrogram, chroma, pitch (pYIN), tempo, beats, RMS, ZCR |
| **Effects** | Reverb, echo, pitch shift, time stretch, EQ, noise reduction, compression |
| **Visualization** | Waveform, spectrogram, MFCC, chroma, pitch, beats — dark theme |
| **Transcription** | Speech-to-text (Whisper), timestamps, multilingual translation |
| **Classification** | Sound events (AudioSet/527 categories), music genre, speech detection |
| **Emotion** | Speech emotion recognition (happy, sad, angry, neutral, ...) |
| **Source Separation** | Vocal isolation, 4-stem separation (drums, bass, vocals, other) |
| **TTS** | Google TTS (fast), SpeechT5 (high quality) |
| **Enhancement** | Deep denoising, speech enhancement pipeline, VAD silence removal |

---

## 📦 Installation

### Quick Install (Core)

```bash
pip install audiopy
```

### Install from Source

```bash
git clone https://github.com/example/audiopy.git
cd audiopy
pip install -e .
```

### Install with Optional Extras

```bash
# Source separation (demucs + spleeter, ~300 MB download)
pip install audiopy[separation]

# WebRTC VAD for silence removal
pip install audiopy[vad]

# DeepFilterNet deep denoising
pip install audiopy[deepfilter]

# SpeechT5 high-quality TTS with speaker embeddings
pip install audiopy[advanced_tts]

# Everything
pip install audiopy[full]
```

### Requirements

- Python >= 3.8
- Core dependencies installed automatically via pip
- GPU (CUDA) recommended but not required for AI models

---

## 🚀 Quick Start

```python
import audiopy as ap

# Load audio
y, sr = ap.load("speech.wav")
print(f"Loaded {ap.duration(y, sr):.2f}s at {sr} Hz")

# Denoise
y_clean = ap.noise_reduce(y, sr)

# Transcribe
text = ap.transcribe(y_clean, sr)
print(f"Transcript: {text}")

# Detect emotion
emotion = ap.dominant_emotion(y_clean, sr)
print(f"Emotion: {emotion}")

# Visualize
fig = ap.plot_waveform(y_clean, sr, title="Cleaned Speech")
ap.save_plot(fig, "waveform.png")

# Save
ap.save("clean.wav", y_clean, sr)
```

---

## 📖 Full API Reference

---

### 🔊 I/O — `ap.io` / top-level

#### `ap.load(path, sr=22050)` → `(np.ndarray, int)`

Load any audio file into a float32 numpy array.

```python
y, sr = ap.load("audio.mp3")           # load and resample to 22050 Hz
y, sr = ap.load("audio.wav", sr=None)  # preserve native sample rate
y, sr = ap.load("audio.flac", sr=44100)
```

**Supported formats:** WAV, MP3, FLAC, OGG, M4A, AIFF and any format supported by librosa.

---

#### `ap.save(path, y, sr, format="wav")`

Save audio to disk.

```python
ap.save("output.wav", y, sr)
ap.save("output.flac", y, sr, format="flac")
ap.save("output.ogg", y, sr, format="ogg")
```

---

#### `ap.record(duration=5, sr=44100)` → `(np.ndarray, int)`

Record from the default microphone.

```python
print("Recording...")
y, sr = ap.record(duration=10, sr=44100)
ap.save("recording.wav", y, sr)
```

---

#### `ap.from_url(url, sr=22050)` → `(np.ndarray, int)`

Download and load audio from any URL.

```python
url = "https://example.com/sample.mp3"
y, sr = ap.from_url(url, sr=16000)
```

---

#### `ap.info(path)` → `dict`

Return audio file metadata without fully loading it.

```python
meta = ap.info("audio.wav")
# {'duration': 3.5, 'sample_rate': 44100, 'channels': 2,
#  'format': 'WAV', 'subtype': 'PCM_16', 'frames': 154350, 'file_size': 617464}
```

---

### ⚙️ Core Processing — `ap.core` / top-level

#### `ap.resample(y, orig_sr, target_sr)` → `np.ndarray`

```python
y_16k = ap.resample(y, orig_sr=44100, target_sr=16000)
```

#### `ap.mono(y)` → `np.ndarray`

Convert stereo to mono by averaging channels.

```python
y_mono = ap.mono(y_stereo)
```

#### `ap.stereo(y)` → `np.ndarray`

Convert mono to stereo by duplicating the channel.

```python
y_stereo = ap.stereo(y_mono)  # shape: (2, N)
```

#### `ap.trim(y, sr, top_db=20)` → `np.ndarray`

Remove leading and trailing silence.

```python
y_trimmed = ap.trim(y, sr, top_db=25)
```

#### `ap.pad(y, target_length, mode="constant")` → `np.ndarray`

Pad or truncate to a fixed number of samples.

```python
y_padded = ap.pad(y, target_length=sr * 10)  # pad to 10 seconds
```

#### `ap.split_on_silence(y, sr, min_silence_len=500, silence_thresh=-40)` → `list[np.ndarray]`

Split audio into non-silent chunks.

```python
segments = ap.split_on_silence(y, sr, min_silence_len=300, silence_thresh=-35)
print(f"Found {len(segments)} speech segments")
```

#### `ap.concatenate(audio_list, sr)` → `np.ndarray`

Concatenate a list of audio arrays.

```python
y_full = ap.concatenate([y1, y2, y3], sr)
```

#### `ap.mix(y1, y2, weight1=0.5, weight2=0.5)` → `np.ndarray`

Mix two signals with optional weighting.

```python
mixed = ap.mix(voice, music, weight1=0.7, weight2=0.3)
```

#### `ap.normalize(y)` → `np.ndarray`

Peak-normalize audio to the range [-1, 1].

```python
y_norm = ap.normalize(y)
```

#### `ap.duration(y, sr)` → `float`

```python
secs = ap.duration(y, sr)
print(f"Duration: {secs:.2f}s")
```

---

### 📊 Feature Extraction — `ap.features` / top-level

#### `ap.mfcc(y, sr, n_mfcc=13)` → `np.ndarray` shape `(n_mfcc, T)`

```python
mfccs = ap.mfcc(y, sr, n_mfcc=40)
```

#### `ap.mel_spectrogram(y, sr, n_mels=128)` → `np.ndarray` shape `(n_mels, T)`

```python
mel = ap.mel_spectrogram(y, sr, n_mels=128)
```

#### `ap.spectrogram(y, sr)` → `np.ndarray` shape `(F, T)`

```python
S = ap.spectrogram(y, sr)
```

#### `ap.chroma(y, sr)` → `np.ndarray` shape `(12, T)`

```python
chroma = ap.chroma(y, sr)
```

#### `ap.zero_crossing_rate(y)` → `np.ndarray` shape `(1, T)`

```python
zcr = ap.zero_crossing_rate(y)
```

#### `ap.spectral_centroid(y, sr)` → `np.ndarray` (Hz)

```python
sc = ap.spectral_centroid(y, sr)
print(f"Mean centroid: {sc.mean():.1f} Hz")
```

#### `ap.spectral_bandwidth(y, sr)` → `np.ndarray` (Hz)

```python
bw = ap.spectral_bandwidth(y, sr)
```

#### `ap.spectral_rolloff(y, sr)` → `np.ndarray` (Hz)

```python
rolloff = ap.spectral_rolloff(y, sr)
```

#### `ap.rms_energy(y)` → `np.ndarray`

```python
rms = ap.rms_energy(y)
```

#### `ap.tempo(y, sr)` → `float` (BPM)

```python
bpm = ap.tempo(y, sr)
print(f"Tempo: {bpm:.1f} BPM")
```

#### `ap.pitch(y, sr)` → `np.ndarray` (Hz, NaN for unvoiced)

```python
f0 = ap.pitch(y, sr)
```

#### `ap.beat_frames(y, sr)` → `np.ndarray` (frame indices)

```python
beats = ap.beat_frames(y, sr)
```

#### `ap.extract_all(y, sr)` → `dict`

Extract every feature in a single call:

```python
features = ap.extract_all(y, sr)
for name, val in features.items():
    print(f"{name}: {val.shape if hasattr(val, 'shape') else val}")
```

---

### 🎛️ Audio Effects — `ap.effects` / top-level

#### `ap.change_pitch(y, sr, steps)` → `np.ndarray`

```python
y_up   = ap.change_pitch(y, sr, steps=4)   # up 4 semitones
y_down = ap.change_pitch(y, sr, steps=-12) # down one octave
```

#### `ap.change_speed(y, rate)` → `np.ndarray`

```python
y_fast = ap.change_speed(y, rate=1.5)  # 50% faster
y_slow = ap.change_speed(y, rate=0.8)  # 20% slower
```

#### `ap.reverb(y, sr, room_scale=0.5, damping=0.5, wet_level=0.3)` → `np.ndarray`

```python
y_reverb = ap.reverb(y, sr, room_scale=0.9, wet_level=0.5)
```

#### `ap.echo(y, sr, delay=0.3, decay=0.4)` → `np.ndarray`

```python
y_echo = ap.echo(y, sr, delay=0.4, decay=0.5)
```

#### `ap.low_pass_filter(y, sr, cutoff=3000)` → `np.ndarray`

```python
y_muffled = ap.low_pass_filter(y, sr, cutoff=1000)
```

#### `ap.high_pass_filter(y, sr, cutoff=300)` → `np.ndarray`

```python
y_clean = ap.high_pass_filter(y, sr, cutoff=80)  # remove rumble
```

#### `ap.bass_boost(y, sr, gain_db=6)` → `np.ndarray`

```python
y_bass = ap.bass_boost(y, sr, gain_db=8)
```

#### `ap.treble_boost(y, sr, gain_db=6)` → `np.ndarray`

```python
y_bright = ap.treble_boost(y, sr, gain_db=4)
```

#### `ap.distortion(y, drive=0.5)` → `np.ndarray`

```python
y_dist = ap.distortion(y, drive=0.7)
```

#### `ap.chorus(y, sr)` → `np.ndarray`

```python
y_chorus = ap.chorus(y, sr)
```

#### `ap.noise_reduce(y, sr, noise_clip=None, prop_decrease=1.0, stationary=False)` → `np.ndarray`

```python
# Auto: use first 0.5s as noise profile
y_clean = ap.noise_reduce(y, sr)

# Manual noise profile
noise, _ = ap.load("noise_sample.wav", sr=sr)
y_clean = ap.noise_reduce(y, sr, noise_clip=noise, stationary=True)
```

#### `ap.noise_reduce_adaptive(y, sr, chunks=4, prop_decrease=0.9)` → `np.ndarray`

Process each segment independently for changing backgrounds:

```python
y_clean = ap.noise_reduce_adaptive(y, sr, chunks=8)
```

#### `ap.compress(y, threshold=-20, ratio=4.0)` → `np.ndarray`

```python
y_compressed = ap.compress(y, threshold=-18, ratio=6.0)
```

#### `ap.fade_in(y, sr, duration=0.5)` → `np.ndarray`

```python
y_faded = ap.fade_in(y, sr, duration=1.0)
```

#### `ap.fade_out(y, sr, duration=0.5)` → `np.ndarray`

```python
y_faded = ap.fade_out(y, sr, duration=2.0)
```

#### `ap.equalize(y, sr, bands)` → `np.ndarray`

3-band parametric EQ with dB gain per band:

```python
y_eq = ap.equalize(y, sr, bands={"low": 4, "mid": -2, "high": 3})
```

---

### 📈 Visualization — `ap.viz` / top-level

All visualization functions return a `matplotlib.figure.Figure` object.
Call `fig.show()` or `ap.save_plot(fig, "file.png")` to display or save.

#### `ap.plot_waveform(y, sr, title="Waveform")`

```python
fig = ap.plot_waveform(y, sr, title="My Recording")
fig.show()
```

#### `ap.plot_spectrogram(y, sr, title="Spectrogram")`

```python
fig = ap.plot_spectrogram(y, sr)
ap.save_plot(fig, "spectrogram.png")
```

#### `ap.plot_mel_spectrogram(y, sr, title="Mel Spectrogram")`

```python
fig = ap.plot_mel_spectrogram(y, sr)
```

#### `ap.plot_mfcc(y, sr, n_mfcc=13)`

```python
fig = ap.plot_mfcc(y, sr, n_mfcc=20)
```

#### `ap.plot_chroma(y, sr)`

```python
fig = ap.plot_chroma(y, sr)
```

#### `ap.plot_pitch(y, sr)`

```python
fig = ap.plot_pitch(y, sr)
```

#### `ap.plot_beats(y, sr)`

```python
fig = ap.plot_beats(y, sr)
```

#### `ap.compare_waveforms(y1, y2, sr, labels=("Original", "Processed"))`

```python
y_noisy, sr = ap.load("noisy.wav")
y_clean = ap.noise_reduce(y_noisy, sr)
fig = ap.compare_waveforms(y_noisy, y_clean, sr, labels=("Noisy", "Denoised"))
```

#### `ap.save_plot(fig, path)`

```python
ap.save_plot(fig, "plots/waveform.png")  # PNG, PDF, SVG supported
```

---

### 🤖 AI Models — `ap.models` / top-level

> **Note:** AI model weights are downloaded automatically on first use.
> Ensure you have an internet connection and sufficient disk space.

---

#### Transcription — Speech-to-Text

##### `ap.transcribe(y, sr, model="openai/whisper-base", language=None)` → `str`

```python
y, sr = ap.load("speech.wav")

# Basic transcription (auto-detects language)
text = ap.transcribe(y, sr)
print(text)

# Force a specific language
text_fr = ap.transcribe(y, sr, language="french")

# Use a larger model for better accuracy
text = ap.transcribe(y, sr, model="openai/whisper-large-v3")
```

**Available Whisper models:** `tiny`, `base`, `small`, `medium`, `large-v3`

##### `ap.transcribe_with_timestamps(y, sr)` → `list[dict]`

```python
segments = ap.transcribe_with_timestamps(y, sr)
for seg in segments:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
```

##### `ap.translate_to_english(y, sr)` → `str`

```python
y, sr = ap.load("spanish_speech.wav")
text_en = ap.translate_to_english(y, sr)
print(text_en)  # English translation
```

---

#### Classification

##### `ap.classify(y, sr, top_k=5)` → `list[dict]`

Classify from 527 AudioSet sound categories:

```python
results = ap.classify(y, sr, top_k=3)
# [{"label": "Dog", "score": 0.97}, {"label": "Animal", "score": 0.02}, ...]
```

##### `ap.is_speech(y, sr)` → `bool`

```python
if ap.is_speech(y, sr):
    text = ap.transcribe(y, sr)
```

##### `ap.classify_music_genre(y, sr)` → `list[dict]`

```python
genres = ap.classify_music_genre(y, sr)
print(f"Genre: {genres[0]['label']} ({genres[0]['score']:.1%})")
```

---

#### Emotion Recognition

##### `ap.detect_emotion(y, sr)` → `list[dict]`

```python
emotions = ap.detect_emotion(y, sr)
for e in emotions:
    print(f"  {e['label']}: {e['score']:.1%}")
```

##### `ap.dominant_emotion(y, sr)` → `str`

```python
emotion = ap.dominant_emotion(y, sr)  # e.g. "happy", "sad", "angry", "neutral"
print(f"Detected emotion: {emotion}")
```

---

#### Source Separation

##### `ap.separate_vocals(y, sr)` → `dict`

Requires `pip install audiopy[separation]` or `pip install demucs`:

```python
y, sr = ap.load("song.mp3", sr=44100)
stems = ap.separate_vocals(y, sr)

ap.save("vocals.wav", stems["vocals"], sr)
ap.save("instrumental.wav", stems["accompaniment"], sr)
```

##### `ap.separate_instruments(y, sr)` → `dict`

4-stem separation (drums, bass, other, vocals):

```python
stems = ap.separate_instruments(y, sr)
ap.save("drums.wav",  stems["drums"],  sr)
ap.save("bass.wav",   stems["bass"],   sr)
ap.save("other.wav",  stems["other"],  sr)
ap.save("vocals.wav", stems["vocals"], sr)
```

---

#### Text-to-Speech

##### `ap.speak(text, language="en", slow=False)` → `(np.ndarray, int)`

```python
y, sr = ap.speak("Hello! This is audiopy.", language="en")
ap.save("tts.wav", y, sr)

# Multilingual
y_fr, sr = ap.speak("Bonjour le monde!", language="fr")
y_ar, sr = ap.speak("مرحباً بالعالم", language="ar")
```

##### `ap.speak_advanced(text, model="microsoft/speecht5_tts")` → `(np.ndarray, int)`

High-quality neural TTS (requires ~1.5 GB download):

```python
y, sr = ap.speak_advanced("High quality speech synthesis with SpeechT5.")
ap.save("hq_tts.wav", y, sr)
```

---

#### Speech Enhancement

##### `ap.denoise_deep(y, sr)` → `np.ndarray`

```python
y_clean = ap.denoise_deep(y, sr)  # Uses DeepFilterNet or noisereduce fallback
```

##### `ap.enhance_speech(y, sr)` → `np.ndarray`

Full speech enhancement pipeline (HPF + denoise + compress + normalize):

```python
y_enhanced = ap.enhance_speech(y, sr)
```

##### `ap.remove_silence(y, sr, aggressiveness=2)` → `np.ndarray`

```python
# aggressiveness: 0 (gentle) to 3 (aggressive)
y_speech_only = ap.remove_silence(y, sr, aggressiveness=2)
print(f"Removed {(len(y)-len(y_speech_only))/sr:.1f}s of silence")
```

---

## 🔧 Utility Functions — `ap.utils`

```python
# Convert between dB and linear
amp = ap.utils.db_to_amplitude(-20.0)     # 0.1
db  = ap.utils.amplitude_to_db(0.5)       # -6.02

# Validate audio
valid = ap.utils.is_valid_audio(y, sr)    # True/False

# Format detection
fmt = ap.utils.detect_format("audio.mp3") # "MP3"

# Time/sample conversion
n = ap.utils.time_to_samples(2.5, 44100)  # 110250
t = ap.utils.samples_to_time(110250, 44100)  # 2.5

# Comprehensive stats
stats = ap.utils.audio_stats(y, sr)
# {'duration_s': 3.5, 'sample_rate': 44100, 'channels': 1,
#  'peak_amplitude': 0.98, 'rms': 0.23, 'peak_db': -0.18, 'rms_db': -12.8, ...}
```

---

## 🧪 Complete Pipeline Example

```python
"""
Full audiopy pipeline:
  Load → Denoise → Transcribe → Detect Emotion → Save + Visualize
"""

import audiopy as ap
import matplotlib.pyplot as plt

# ── 1. Load audio ────────────────────────────────────────────────────────────
print("Loading audio...")
y, sr = ap.load("noisy_speech.wav")
print(f"  Duration: {ap.duration(y, sr):.2f}s | Sample Rate: {sr} Hz")
print(f"  Stats: {ap.utils.audio_stats(y, sr)}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
print("\nPreprocessing...")
y = ap.mono(y)                          # Ensure mono
y = ap.trim(y, sr, top_db=20)          # Remove leading/trailing silence
y = ap.normalize(y)                     # Peak normalize

# ── 3. Noise Reduction ───────────────────────────────────────────────────────
print("\nReducing noise...")
y_clean = ap.noise_reduce(y, sr)
y_clean = ap.enhance_speech(y_clean, sr)  # Full enhancement pipeline

# ── 4. Feature Extraction ────────────────────────────────────────────────────
print("\nExtracting features...")
mfccs = ap.mfcc(y_clean, sr, n_mfcc=13)
bpm   = ap.tempo(y_clean, sr)
print(f"  Estimated tempo: {bpm:.1f} BPM")

# ── 5. AI Analysis ───────────────────────────────────────────────────────────
print("\nRunning AI analysis...")

# Speech-to-text
transcript = ap.transcribe(y_clean, sr)
print(f"  Transcript: {transcript}")

# Emotion detection
emotions = ap.detect_emotion(y_clean, sr)
top_emotion = emotions[0]
print(f"  Dominant emotion: {top_emotion['label']} ({top_emotion['score']:.1%})")

# Is it speech?
print(f"  Is speech: {ap.is_speech(y_clean, sr)}")

# ── 6. Apply Effects (optional) ──────────────────────────────────────────────
y_eq = ap.equalize(y_clean, sr, bands={"low": 2, "mid": 0, "high": 1})
y_final = ap.fade_in(y_eq, sr, duration=0.1)
y_final = ap.fade_out(y_final, sr, duration=0.2)

# ── 7. Visualize ─────────────────────────────────────────────────────────────
print("\nGenerating visualizations...")
fig_compare = ap.compare_waveforms(y, y_final, sr,
                                    labels=("Raw", "Enhanced + EQ"))
ap.save_plot(fig_compare, "comparison.png")

fig_mel = ap.plot_mel_spectrogram(y_final, sr)
ap.save_plot(fig_mel, "mel_spectrogram.png")

fig_mfcc = ap.plot_mfcc(y_final, sr)
ap.save_plot(fig_mfcc, "mfcc.png")

# ── 8. Save ───────────────────────────────────────────────────────────────────
print("\nSaving output...")
ap.save("output_clean.wav", y_final, sr)
print("Done! 🎵")
```

---

## 🏗️ Package Structure

```
audiopy/
├── __init__.py          # Top-level API — import everything from here
├── io.py                # Audio I/O: load, save, record, from_url, info
├── core.py              # Processing: resample, trim, pad, mix, normalize
├── features.py          # Feature extraction: MFCC, spectrogram, pitch, tempo
├── effects.py           # DSP effects: reverb, EQ, noise reduction, filters
├── viz.py               # Visualization: waveform, spectrogram, MFCC plots
├── utils.py             # Utilities: dB, stats, format detection, conversions
├── models/
│   ├── __init__.py      # Models namespace
│   ├── transcribe.py    # Whisper speech-to-text
│   ├── classifier.py    # AST audio classification, genre, speech detection
│   ├── emotion.py       # Wav2Vec2 speech emotion recognition
│   ├── separation.py    # Demucs source separation (vocals, stems)
│   ├── tts.py           # gTTS + SpeechT5 text-to-speech
│   └── enhancement.py   # DeepFilterNet + noisereduce + WebRTC VAD
├── setup.py             # Package installer
└── requirements.txt     # Core dependencies
```

---

## 🧩 Dependency Overview

| Package | Purpose | Required |
|---|---|---|
| `librosa` | Audio loading, resampling, features | ✅ Core |
| `soundfile` | Audio file I/O | ✅ Core |
| `sounddevice` | Microphone recording | ✅ Core |
| `numpy` | Array processing | ✅ Core |
| `matplotlib` | Visualization | ✅ Core |
| `scipy` | Signal processing, filters | ✅ Core |
| `transformers` | Whisper, AST, Wav2Vec2, SpeechT5 | ✅ Core |
| `torch` | Deep learning backend | ✅ Core |
| `torchaudio` | Audio tensor ops | ✅ Core |
| `noisereduce` | Spectral noise reduction | ✅ Core |
| `pedalboard` | Reverb, chorus effects | ✅ Core |
| `gTTS` | Google TTS | ✅ Core |
| `requests` | URL audio download | ✅ Core |
| `demucs` | Source separation | Optional (`[separation]`) |
| `spleeter` | Source separation fallback | Optional (`[separation]`) |
| `webrtcvad` | Voice activity detection | Optional (`[vad]`) |
| `deepfilternet` | Deep noise removal | Optional (`[deepfilter]`) |
| `datasets` | SpeechT5 speaker embeddings | Optional (`[advanced_tts]`) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

```bash
# Development install
pip install -e "audiopy[dev]"

# Run tests
pytest tests/ -v
```

---

*Built with ❤️ using librosa, HuggingFace Transformers, PyTorch, and pedalboard.*
