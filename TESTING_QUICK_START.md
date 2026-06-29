# 🎬 audiopy_ai AI Analysis - Quick Start Testing Guide

## ⚡ Get Started in 5 Minutes

### Step 1: Install FFmpeg (First time only)

**Windows:**
```powershell
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

### Step 2: Install Python Dependencies

```bash
pip install transformers torch librosa
```

### Step 3: Test Your Setup

**Option A: Quick Test** (Recommended)
```bash
python quick_test.py
```

**Option B: Interactive Tool** (User-friendly)
```bash
python interactive_test.py
```

**Option C: Test with Your MP4** (Direct approach)
```bash
python test_mp4_analysis.py your_video.mp4
```

## 📊 Three Ways to Test

### 1. 🚀 Fastest: Interactive CLI Tool
```bash
python interactive_test.py
```

**Features:**
- User-friendly menu system
- Upload MP4 or audio files
- Test with synthetic audio
- Change analysis types on the fly
- View previous results

**Steps:**
1. Select "1️⃣ Analyze MP4/Video File"
2. Enter your file path
3. Wait for analysis
4. View results with confidence scores

### 2. 🎯 Direct: MP4 Analysis Script
```bash
python test_mp4_analysis.py video.mp4
```

**Pros:**
- Single command
- Results saved to JSON
- No additional prompts

**Example:**
```bash
python test_mp4_analysis.py "C:\Videos\sample.mp4"
```

### 3. ✅ Verify: Test Suite
```bash
python tests/test_core.py
```

**For comprehensive testing of all components.**

## 📋 What Each Tool Does

| Tool | Purpose | Use When |
|------|---------|----------|
| `quick_test.py` | Verify installation | First time setup |
| `test_mp4_analysis.py` | Analyze MP4 files | Test with specific videos |
| `interactive_test.py` | Interactive testing | Want menu-driven approach |
| `tests/test_core.py` | Run unit tests | Want comprehensive validation |

## 🎥 Test with Your MP4 File

### Simple Command
```bash
python test_mp4_analysis.py "path/to/your/video.mp4"
```

### Example Output
```
======================================================================
audiopy_ai AI Analysis - MP4 Audio Test
======================================================================

📁 Input File: video.mp4
📊 File Size: 150.25 MB
🔍 Analysis Type: audio_tagging

Step 1️⃣  Extracting audio from MP4...
   ✅ Audio extracted: 45.67 MB

Step 2️⃣  Loading audio file...
   ✅ Loaded: 2,000,000 samples at 16000Hz
   📈 Duration: 125.00 seconds

Step 3️⃣  Initializing AI Analyzer...
   ✅ Analyzer ready

Step 4️⃣  Analyzing audio...
   ⏳ Processing...
   ✅ Analysis complete!

======================================================================
📊 ANALYSIS RESULTS
======================================================================

🔊 DETECTED SOUND COMPONENTS:
----------------------------------------------------------------------
  1. dog                         [████████████████████░░░░░░░░░░] 92.3%
  2. bark                        [████████████       ░░░░░░░░░░░░] 75.6%
  3. animal                      [████████           ░░░░░░░░░░░░░░] 54.8%

📈 AUDIO FEATURES:
----------------------------------------------------------------------
  Spectral Centroid: 3245.67 Hz
  Spectral Rolloff:  8234.23 Hz
  Zero Crossing Rate: 0.1234

✅ Results saved to: video_analysis.json
```

## 🎯 Different Analysis Types

### Audio Tagging (Default - Recommended)
Detects sound events:
```bash
python test_mp4_analysis.py video.mp4
```

### Speech Recognition
Transcribes speech:
```python
from audiopy_ai.ai_analysis import SoundComponentExtractor
analyzer = SoundComponentExtractor(analysis_type="speech_recognition")
results = analyzer.extract_components("audio.wav")
```

### Emotion Detection  
Detects emotions from speech:
```python
analyzer = SoundComponentExtractor(analysis_type="emotion_detection")
```

### Music Analysis
Analyzes music characteristics:
```python
analyzer = SoundComponentExtractor(analysis_type="music_analysis")
```

## 🔍 Understanding Results

### Confidence Scores
- 90-100%: Very confident detection
- 70-89%: Good confidence
- 50-69%: Moderate confidence
- <50%: Low confidence

### Audio Features Explained
| Feature | What It Means |
|---------|--------------|
| Spectral Centroid | Brightness of sound (Hz) |
| Spectral Rolloff | High-frequency cutoff (Hz) |
| Zero Crossing Rate | Noisiness indicator (0-1) |
| Onset Strength | Rhythm/beat prominence |
| MFCC | Speech/music fingerprint |

## 📁 Output Files

After each analysis, a JSON file is created:
- **Filename:** `video_name_analysis.json`
- **Location:** Same directory as the script
- **Contents:** Full analysis results

### Example JSON Structure
```json
{
  "model": "mtg-upf/discogs-effnet-bs64-1",
  "task": "audio-classification",
  "components": {
    "detected_sounds": ["dog", "bark"],
    "confidence_scores": [0.923, 0.756],
    "metadata": {}
  },
  "audio_features": {
    "spectral_centroid": 3245.67,
    "mfcc_mean": [...]
  }
}
```

## ⚠️ Common Issues & Solutions

### "ffmpeg was not found"
```bash
# Windows
winget install ffmpeg

# Then restart your terminal
```

### "No module named 'transformers'"
```bash
pip install transformers torch
```

### "CUDA out of memory" (GPU error)
Use CPU instead or smaller model:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Slow first run (downloading models)
This is normal! Models download on first use:
- First run: 2-5 minutes (500MB+ download)
- Subsequent runs: A few seconds

### File format not supported
Ensure you're using a supported format: MP4, AVI, MOV, WAV, MP3, FLAC

## 🚀 Performance Tips

1. **GPU Support:** Install CUDA for 10-50x speedup
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Smaller Models:** Use for faster processing on CPU
   ```python
   from audiopy_ai.ai_analysis import AudioAnalyzer
   analyzer = AudioAnalyzer(model_name="facebook/wav2vec2-base")
   ```

3. **Caching:** Models are cached after first download

## 📚 Learn More

- **Setup:** [SETUP.md](SETUP.md) - Full installation guide
- **Guide:** [AI_ANALYSIS_GUIDE.md](AI_ANALYSIS_GUIDE.md) - Detailed API docs
- **Examples:** [examples/ai_analysis_examples.py](examples/ai_analysis_examples.py) - Code samples
- **Testing:** [TESTING.md](TESTING.md) - More test options

## ✅ Checklist

- [ ] FFmpeg installed (`ffmpeg -version` works)
- [ ] Python dependencies installed (`pip list` shows transformers, torch)
- [ ] Quick test passes (`python quick_test.py`)
- [ ] Can analyze test audio without errors
- [ ] Ready to test with your MP4 files!

## 🎬 Ready to Go!

You're all set! Pick your testing method:

**Fastest:**
```bash
python interactive_test.py
```

**With Your File:**
```bash
python test_mp4_analysis.py "C:\path\to\video.mp4"
```

**Quick Verify:**
```bash
python quick_test.py
```

---

**Questions?** Check the documentation files or run the diagnostic test:
```bash
python interactive_test.py  # Select option 6
```
