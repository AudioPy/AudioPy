# 🎵 audiopy_ai - AI-Powered Audio Analysis Library

audiopy_ai is a Python library for audio processing with AI-powered sound component analysis using HuggingFace models. Just like OpenCV for images, audiopy_ai is a comprehensive library for working with audio.

## ✨ Features

### Core Audio Processing
- **I/O**: Load and save audio files (WAV, MP3, FLAC, etc.)
- **Editing**: Trim, fade, normalize, apply gain, concatenate, reverse
- **Effects**: Filters, reverb, time-based effects (future)

### AI-Powered Analysis
- **Audio Tagging**: Detect sound events (dogs, music, speech, etc.)
- **Speech Recognition**: Convert speech to text
- **Emotion Detection**: Analyze emotions from speech
- **Music Analysis**: Extract music characteristics
- **Audio Features**: Spectral analysis, MFCC, onset detection

## 🚀 Quick Start

### 1. Install
```bash
# Install FFmpeg (required)
winget install ffmpeg  # Windows

# Install Python packages
pip install transformers torch librosa
```

### 2. Test Immediately
```bash
# Simple test
python quick_test.py

# Or interactive tool
python interactive_test.py

# Or with your MP4 file
python test_mp4_analysis.py video.mp4
```

### 3. Use in Your Code
```python
from audiopy_ai.ai_analysis import SoundComponentExtractor
from audiopy_ai.io import load

# Analyze an audio file
analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
results = analyzer.extract_components("audio.wav")

# Get detected sounds
sounds = results['components']['detected_sounds']
scores = results['components']['confidence_scores']

for sound, score in zip(sounds, scores):
    print(f"{sound}: {score:.1%}")
```

## 📊 Testing

<details>
<summary><b>Click to see all testing options</b></summary>

### Interactive Tool (Recommended for first-time users)
```bash
python interactive_test.py
```
- User-friendly menu system
- Upload MP4 or audio files
- Test with synthetic audio
- Change analysis types
- View previous results

### Direct MP4 Analysis
```bash
python test_mp4_analysis.py video.mp4
```
- Quick test with your video
- Results saved to JSON
- No menu prompts

### Quick System Verification
```bash
python quick_test.py
```
- Verify installation
- Test I/O and editing
- Basic functionality check

### Full Test Suite
```bash
python tests/test_core.py
```
- Comprehensive unit tests
- All core functionality
- Error handling

### See Results
All tools save results to `video_name_analysis.json` or similar.

</details>

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [TESTING_QUICK_START.md](TESTING_QUICK_START.md) | How to test with MP4/audio files |
| [SETUP.md](SETUP.md) | Installation and troubleshooting |
| [AI_ANALYSIS_GUIDE.md](AI_ANALYSIS_GUIDE.md) | API reference and examples |
| [TESTING.md](TESTING.md) | Testing options and validation |

## 📚 Examples

### Basic Audio Analysis
```python
from audiopy_ai.ai_analysis import SoundComponentExtractor

analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
results = analyzer.extract_components("audio.wav")
```

### With Audio Features
```python
from audiopy_ai.io import load

samples, sr = load("audio.wav")
results = analyzer.extract_components_with_features(
    "audio.wav", samples, sr
)

# Access features
print(f"Spectral Centroid: {results['audio_features']['spectral_centroid']} Hz")
```

### MP4 Processing
```bash
python test_mp4_analysis.py "C:\Videos\sample.mp4"
# Extracts audio, analyzes, saves results to sample_analysis.json
```

### Different Analysis Types
```python
# Speech to text
analyzer = SoundComponentExtractor(analysis_type="speech_recognition")

# Emotion detection
analyzer = SoundComponentExtractor(analysis_type="emotion_detection")

# Music analysis
analyzer = SoundComponentExtractor(analysis_type="music_analysis")
```

See [examples/ai_analysis_examples.py](examples/ai_analysis_examples.py) for more.

## 🎯 Supported Formats

### Audio
✅ WAV, MP3, FLAC, OGG, M4A, WMA

### Video
✅ MP4, AVI, MOV, MKV, WMV, WebM, 3GP

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| Python | 3.9 | 3.10+ |
| RAM | 4GB | 8GB+ |
| Disk | 1GB | 5GB+ |
| GPU | Optional | NVIDIA CUDA |

## 📦 Project Structure

```
audiopy_ai/
├── audiopy_ai/
│   ├── __init__.py
│   ├── ai_analysis.py        # AI-powered analysis
│   ├── io.py                 # Load/save audio
│   ├── edit.py               # Audio editing
│   └── effects/              # Audio effects
├── examples/
│   └── ai_analysis_examples.py
├── tests/
│   ├── test_core.py
│   ├── test_ai_analysis.py
│   └── test_io.py
├── interactive_test.py       # Interactive testing tool
├── test_mp4_analysis.py      # MP4 analysis tool
├── quick_test.py             # Quick verification
├── SETUP.md                  # Installation guide
├── TESTING_QUICK_START.md    # Testing guide
├── AI_ANALYSIS_GUIDE.md      # API reference
└── README.md                 # This file
```

## 🔧 Troubleshooting

### "ffmpeg was not found"
```bash
winget install ffmpeg  # Windows
brew install ffmpeg    # Mac
sudo apt-get install ffmpeg  # Linux
```

### "No module named 'transformers'"
```bash
pip install transformers torch
```

### Slow performance
- First run downloads models (2-5 mins) - normal!
- Install GPU support for 10-50x speedup
- Use smaller models for CPU-only

## 📊 Example Output

```
🔊 DETECTED SOUND COMPONENTS:
  1. dog                   [██████████████████░░░░] 92.3%
  2. bark                  [████████████░░░░░░░░░░░] 76.5%
  3. animal                [█████████░░░░░░░░░░░░░░░] 54.1%

📈 AUDIO FEATURES:
  Spectral Centroid: 3245.67 Hz
  Spectral Rolloff:  8234.23 Hz
  Zero Crossing:     0.1234
```

## 🚀 Performance

- **First run:** 2-5 minutes (models download)
- **Subsequent:** A few seconds per file
- **GPU:** 10-50x faster with CUDA
- **Models cached:** Instant after first download

## 📝 License

MIT License - See LICENSE folder

## 🤝 Contributing

Contributions welcome! Check the examples and tests for patterns.

## 📞 Support

1. Check [SETUP.md](SETUP.md) for installation issues
2. Review [TESTING_QUICK_START.md](TESTING_QUICK_START.md) for testing
3. See [AI_ANALYSIS_GUIDE.md](AI_ANALYSIS_GUIDE.md) for API docs
4. Run diagnostic: `python interactive_test.py` (option 6)

## 🎬 Get Started Now

```bash
# Install and test in 2 steps:
pip install transformers torch librosa
python interactive_test.py
```

Then select option 1️⃣ to analyze your first MP4 file!

---

**Latest Update:** February 2026 | **Version:** 0.1.3
