# audiopy_ai Setup Guide

## System Requirements

### FFmpeg (Required)

audiopy_ai uses `soundfile` which requires FFmpeg to load audio files.

#### Windows Installation

**Option 1: Using Windows Package Manager (Recommended)**
```powershell
winget install FFmpeg
```

**Option 2: Using Chocolatey**
```powershell
choco install ffmpeg
```

**Option 3: Manual Installation**
1. Download from: https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH:
   - Open Environment Variables
   - Add `C:\ffmpeg\bin` to System PATH
   - Restart your terminal

**Verify Installation:**
```bash
ffmpeg -version
ffprobe -version
```

#### macOS Installation

```bash
# Using Homebrew
brew install ffmpeg
```

#### Linux Installation

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Fedora
sudo dnf install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

### Python Dependencies

#### Minimal Installation (I/O and Editing Only)
```bash
pip install numpy scipy soundfile
```

#### Full Installation (With AI Analysis)
```bash
pip install numpy scipy soundfile transformers torch librosa
```

#### Development Installation
```bash
pip install -e .
```

## Installation Steps

### 1. Install FFmpeg (System-wide)
See instructions above for your OS.

### 2. Clone or Download audiopy_ai
```bash
cd c:\Users\LALIT\Documents\Codes\projects\AudioPy
```

### 3. Install audiopy_ai and Dependencies
```bash
# Option A: Minimal
pip install numpy scipy soundfile

# Option B: Full (includes AI)
pip install numpy scipy soundfile transformers torch librosa

# Option C: From project (installs what's in pyproject.toml)
pip install -e .
```

### 4. Verify Installation
```bash
python quick_test.py
```

## Troubleshooting

### Error: "ffmpeg was not found"
**Solution:** Install FFmpeg (see above)

**Verify:**
```bash
ffmpeg -version
```

### Error: "No module named 'transformers'"
**Solution:** Install AI dependencies
```bash
pip install transformers torch
```

### Error: "librosa not installed"
**Optional:** Install librosa for audio feature extraction
```bash
pip install librosa
```

### Error: "CUDA out of memory" (GPU errors)
**Solution:** Use a smaller model
```python
from audiopy_ai.ai_analysis import AudioAnalyzer
analyzer = AudioAnalyzer(model_name="facebook/wav2vec2-base")
```

Or use CPU only:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Complete Setup Commands

### Windows (PowerShell)
```powershell
# 1. Install FFmpeg
winget install FFmpeg

# 2. Close and reopen PowerShell

# 3. Install Python dependencies
pip install numpy scipy soundfile transformers torch librosa

# 4. Navigate to audiopy_ai
cd c:\Users\LALIT\Documents\Codes\projects\AudioPy

# 5. Test installation
python quick_test.py

# 6. Run comprehensive tests
python tests/test_core.py
```

### macOS/Linux (Bash)
```bash
# 1. Install FFmpeg
brew install ffmpeg  # macOS
# or
sudo apt-get install ffmpeg  # Ubuntu/Debian

# 2. Install Python dependencies
pip install numpy scipy soundfile transformers torch librosa

# 3. Navigate to AudioPy
cd ~/path/to/AudioPy

# 4. Test installation
python quick_test.py

# 5. Run comprehensive tests
python tests/test_core.py
```

## Verify Everything Works

After installation, run:

```bash
python quick_test.py
```

Expected output:
```
============================================================
audiopy_ai AI Analysis - Quick Test
============================================================

1️⃣  Creating test audio...
   ✓ Created 32000 samples at 16000Hz

2️⃣  Saving audio file...
   ✓ Saved to test.wav

3️⃣  Loading and processing audio...
   ✓ Processed: normalized + fade

4️⃣  Initializing AI Analyzer...
   ✓ Analyzer ready

5️⃣  Analyzing audio file...
   ✓ Analysis complete!

   Detected Components:
     • [sounds detected]

============================================================
✅ ALL TESTS PASSED!
```

## Environment Variables (Optional)

### Use CPU Only (No GPU)
```bash
# PowerShell
$env:CUDA_VISIBLE_DEVICES = '-1'
python quick_test.py

# Bash
export CUDA_VISIBLE_DEVICES='-1'
python quick_test.py
```

## Next Steps

After successful installation:

1. **Read the Quick Reference:** [AI_ANALYSIS_GUIDE.md](AI_ANALYSIS_GUIDE.md)
2. **Run Examples:** [examples/ai_analysis_examples.py](examples/ai_analysis_examples.py)
3. **Read Testing Guide:** [TESTING.md](TESTING.md)
4. **Test with MP4 Files:** [test_mp4_analysis.py](test_mp4_analysis.py)
5. **Integrate into Your Code:** See examples for usage patterns

## Testing with MP4 Files

audiopy_ai can analyze audio extracted from MP4 video files.

### Quick Test (Creates Synthetic Audio)
```bash
python test_mp4_analysis.py
```

### Test with Your MP4 File
```bash
python test_mp4_analysis.py your_video.mp4
```

**Example:**
```bash
python test_mp4_analysis.py sample_video.mp4
```

### Expected Output

```
======================================================================
audiopy_ai AI Analysis - MP4 Audio Test
======================================================================

📁 Input File: sample_video.mp4
📊 File Size: 45.32 MB
🔍 Analysis Type: audio_tagging

Step 1️⃣  Extracting audio from MP4...
   ✅ Audio extracted: 12.45 MB

Step 2️⃣  Loading audio file...
   ✅ Loaded: 2,048,000 samples at 16000Hz
   📈 Duration: 128.00 seconds

Step 3️⃣  Initializing AI Analyzer...
   ✅ Analyzer ready

Step 4️⃣  Analyzing audio...
   ⏳ Processing (this may take a moment on first run)...
   ✅ Analysis complete!

======================================================================
📊 ANALYSIS RESULTS
======================================================================

🔊 DETECTED SOUND COMPONENTS:
----------------------------------------------------------------------
  1. dog                         [██████████████████████████░░░░░░░░░░] 92.3%
  2. bark                        [████████████████████░░░░░░░░░░░░░░░░░░] 76.5%
  3. sound                       [█████████████░░░░░░░░░░░░░░░░░░░░░░░░░] 58.2%

📈 AUDIO FEATURES:
----------------------------------------------------------------------
  Spectral Centroid (brightness): 3245.67 Hz
  Spectral Rolloff:               8234.23 Hz
  Zero Crossing Rate (noisiness): 0.1234

✅ Results saved to: sample_video_analysis.json
```

### Supported Video Formats
- MP4, AVI, MOV, MKV, WMV, WebM, 3GP, FLV

### Supported Audio Formats  
- WAV, MP3, FLAC, OGG, M4A, WMA

### Output
Analysis results are saved to `<video_name>_analysis.json`

Example output file:
```json
{
  "model": "mtg-upf/discogs-effnet-bs64-1",
  "task": "audio-classification",
  "components": {
    "detected_sounds": ["dog", "bark", "animal"],
    "confidence_scores": [0.923, 0.765, 0.541]
  },
  "audio_features": {
    "spectral_centroid": 3245.67,
    "mfcc_mean": [...]
  }
}
```

## Support

If you encounter issues:

1. Check [TESTING.md](TESTING.md) for common issues
2. Verify FFmpeg is installed: `ffmpeg -version`
3. Verify Python packages: `pip list`
4. Run the test suite: `python tests/test_core.py`
5. Test MP4 analysis: `python test_mp4_analysis.py`
