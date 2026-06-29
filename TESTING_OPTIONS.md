# 🧪 audiopy_ai AI Analysis - Complete Testing Options

## Overview

audiopy_ai provides multiple ways to test your AI model with audio and video files:

| Tool | File | Type | Best For | Command |
|------|------|------|----------|---------|
| 🎯 Quick Test | `quick_test.py` | Verification | Verify installation | `python quick_test.py` |
| 🎬 MP4 Analyzer | `test_mp4_analysis.py` | Direct | Test MP4 files | `python test_mp4_analysis.py video.mp4` |
| 🖥️ Interactive | `interactive_test.py` | Menu-driven | User-friendly testing | `python interactive_test.py` |
| 📖 Examples | `simple_usage_example.py` | Demo | Learn usage patterns | `python simple_usage_example.py` |
| ✅ Unit Tests | `tests/test_core.py` | Validation | Comprehensive testing | `python tests/test_core.py` |

## 🚀 Choose Your Testing Method

### Method 1: Interactive Tool (RECOMMENDED FOR FIRST TIME)

**Best for:** Users who want a guided experience

```bash
python interactive_test.py
```

**What you can do:**
- 1️⃣ Analyze MP4/Video files
- 2️⃣ Analyze audio files (WAV, MP3, etc.)
- 3️⃣ Test with synthetic audio
- 4️⃣ Change analysis types
- 5️⃣ View previous results
- 6️⃣ Run diagnostics

**Pros:**
- User-friendly menus
- No command-line syntax needed
- Complete workflow
- Built-in diagnostics

**Cons:**
- Requires user interaction
- Slower than direct commands

### Method 2: Direct MP4 Analysis (FASTEST)

**Best for:** Users who just want to test their MP4 file

```bash
python test_mp4_analysis.py "C:\path\to\video.mp4"
```

**What it does:**
1. Extracts audio from MP4
2. Loads and analyzes
3. Saves results to JSON
4. Displays results

**Pros:**
- Single command
- Fast
- Automatic JSON output
- No interaction needed

**Cons:**
- No menu options
- Requires file path as argument

**Example:**
```bash
python test_mp4_analysis.py "C:\Videos\sample.mp4"
# Creates: sample_analysis.json
```

### Method 3: Quick Test (VERIFY INSTALLATION)

**Best for:** First-time setup verification

```bash
python quick_test.py
```

**What it tests:**
- ✓ Audio file I/O
- ✓ Audio processing
- ✓ AI analyzer initialization
- ✓ Basic analysis

**Expected output:**
```
✅ ALL TESTS PASSED!
✅ Core I/O and editing working!
```

### Method 4: Simple Examples (LEARN USAGE)

**Best for:** Understanding how to use AudioPy

```bash
python simple_usage_example.py
```

**Includes:**
1. Basic analysis
2. MP4 analysis instructions
3. Analysis with features
4. Different analysis types
5. Editing and analysis

### Method 5: Unit Tests (COMPREHENSIVE VALIDATION)

**Best for:** Developers and CI/CD

```bash
python tests/test_core.py
```

**Tests:**
- Audio I/O operations
- Editing functions
- AI module imports
- Integration pipeline
- Error handling

## 📊 Using MP4 Files - Step by Step

### Step 1: Prepare Your MP4
You need:
- Valid MP4 file
- Audio track (mono or stereo)
- Supported codec (H.264 video, AAC audio)

**Example files:**
- `video.mp4` - From phone/camera
- `tutorial.mp4` - Downloaded video
- `presentation.mp4` - Screen recording

### Step 2: Run Analysis

#### Option A: Direct Command (Recommended)
```bash
python test_mp4_analysis.py video.mp4
```

#### Option B: Interactive Tool
```bash
python interactive_test.py
# Select: 1 → Enter file path → Done
```

#### Option C: In Python Code
```python
import tempfile
import subprocess
from audiopy_ai.io import load
from audiopy_ai.ai_analysis import SoundComponentExtractor

# Extract audio from MP4
with tempfile.TemporaryDirectory() as tmpdir:
    wav_path = f"{tmpdir}/audio.wav"
    
    # Use ffmpeg to extract
    subprocess.run([
        'ffmpeg', '-i', 'video.mp4',
        '-q:a', '9', '-n', wav_path
    ])
    
    # Analyze
    analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
    results = analyzer.extract_components(wav_path)
    print(results)
```

### Step 3: View Results

**Console Output:**
```
🔊 DETECTED SOUND COMPONENTS:
  1. dog                   [████████████░░░░░░░░░░░░] 92.3%
  2. bark                  [████████░░░░░░░░░░░░░░░░░] 76.5%

📈 AUDIO FEATURES:
  Spectral Centroid: 3245.67 Hz
```

**JSON File:**
- Name: `video_analysis.json`
- Location: Current directory
- Contains: Full analyis results

## 🎯 Analysis Types You Can Test

### 1. Audio Tagging (Default)
**Detects sound events:**
- Dogs barking
- Music playing
- Speech
- Traffic
- Applause
- Various sounds

```bash
python test_mp4_analysis.py video.mp4
```

### 2. Speech Recognition
**Transcribes speech to text:**

```python
analyzer = SoundComponentExtractor(analysis_type="speech_recognition")
results = analyzer.extract_components("audio.wav")
transcription = results['components']['metadata']['transcription']
```

### 3. Emotion Detection
**Detects emotions from speech:**

```python
analyzer = SoundComponentExtractor(analysis_type="emotion_detection")
results = analyzer.extract_components("audio.wav")
emotions = results['components']['detected_sounds']
```

### 4. Music Analysis
**Analyzes music characteristics:**

```python
analyzer = SoundComponentExtractor(analysis_type="music_analysis")
results = analyzer.extract_components("music.wav")
```

## 📋 Interpreting Results

### Confidence Scores
```
95%+ - Excellent confidence (definitely detected)
80-94% - Very good confidence
60-79% - Good confidence
40-59% - Moderate confidence
<40% - Low confidence (may be false positive)
```

### Audio Features
```
Spectral Centroid: 0-22050 Hz (brightness of sound)
Spectral Rolloff: 0-22050 Hz (high-freq cutoff)
Zero Crossing: 0.0-1.0 (noisiness, higher = noisier)
Onset Strength: 0.0+ (rhythm prominence)
MFCC: 13 values (speech/music fingerprint)
```

## 🔧 Troubleshooting

### Problem: "ffmpeg was not found"
```bash
# Solution:
winget install ffmpeg  # Windows
brew install ffmpeg    # macOS
sudo apt-get install ffmpeg  # Linux
```

### Problem: "transformers not found"
```bash
pip install transformers torch
```

### Problem: First run is slow
This is **normal**. Models download on first use (500MB+).
- First run: 2-5 minutes
- Subsequent runs: A few seconds

### Problem: CUDA out of memory (GPU)
```bash
# Use CPU instead:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Or use smaller model:
from audiopy_ai.ai_analysis import AudioAnalyzer
analyzer = AudioAnalyzer(model_name="facebook/wav2vec2-base")
```

### Problem: File format not supported
Make sure the file is in a supported format:
- Audio: WAV, MP3, FLAC, OGG, M4A, WMA
- Video: MP4, AVI, MOV, MKV, WMV, WebM, 3GP

## 📊 Output Formats

### Console Output
```
======================================================================
audiopy_ai AI Analysis - MP4 Audio Test
======================================================================

📁 Input File: video.mp4
📊 File Size: 150.25 MB
🔍 Analysis Type: audio_tagging

🔊 DETECTED SOUND COMPONENTS:
  1. dog                   [████████░░░░░░░░░░░░░░░░] 92.3%
  
📈 AUDIO FEATURES:
  Spectral Centroid: 3245.67 Hz
```

### JSON Output (video_analysis.json)
```json
{
  "model": "mtg-upf/discogs-effnet-bs64-1",
  "task": "audio-classification",
  "components": {
    "detected_sounds": ["dog", "bark"],
    "confidence_scores": [0.923, 0.765],
    "metadata": {}
  },
  "audio_features": {
    "spectral_centroid": 3245.67,
    "mfcc_mean": [...]
  }
}
```

## 🚀 Performance Tips

### Speed Up Processing
1. **First run caching:** Models download once
2. **Use GPU:** Install CUDA for 10-50x speedup
3. **Smaller models:** Faster on CPU
4. **Process in chunks:** For very long files

### Reduce Resource Usage
1. **Use CPU only:** Saves GPU memory
2. **Smaller model**: Reduces memory footprint
3. **Process in batches:** Don't load entire file

## 📚 Learning Resources

1. **Quick Start:** [TESTING_QUICK_START.md](TESTING_QUICK_START.md)
2. **Setup Guide:** [SETUP.md](SETUP.md)
3. **API Reference:** [AI_ANALYSIS_GUIDE.md](AI_ANALYSIS_GUIDE.md)
4. **Complete Examples:** [examples/ai_analysis_examples.py](examples/ai_analysis_examples.py)

## ✅ Quick Checklist

Before testing:
- [ ] FFmpeg installed (`ffmpeg -version`)
- [ ] Python packages installed (`pip list | grep transformers`)
- [ ] Have an MP4 or audio file ready
- [ ] At least 4GB RAM available
- [ ] Internet connection (for first model download)

Testing:
- [ ] Run `python quick_test.py` (verify)
- [ ] Run `python test_mp4_analysis.py video.mp4` (test)
- [ ] Check results in console and JSON file

## 🎬 You're Ready!

Pick your method and test:

```bash
# Interactive (easiest)
python interactive_test.py

# Direct with your MP4 (fastest)
python test_mp4_analysis.py "C:\path\to\video.mp4"

# Learn how to use (educational)
python simple_usage_example.py

# Verify installation (quick check)
python quick_test.py
```

---

**Need help?** Check [SETUP.md](SETUP.md) or run the diagnostic in interactive tool (option 6).
