# Testing audiopy_ai AI Analysis

## Quick Start Test (Recommended)

Run the quickest test to verify everything is working:

```bash
cd c:\Users\LALIT\Documents\Codes\projects\AudioPy
python quick_test.py
```

**What it does:**
- ✓ Creates sample audio
- ✓ Tests I/O (save/load)
- ✓ Tests editing (normalize, fade)
- ✓ Initializes AI analyzer
- ✓ Tests audio analysis

**Expected Output:**
```
============================================================
audiopy_ai AI Analysis - Quick Test
============================================================

1️⃣  Creating test audio...
   ✓ Created 32000 samples at 16000Hz

2️⃣  Saving audio file...
   ✓ Saved to test.wav

... [continues] ...

✅ ALL TESTS PASSED!
```

## Comprehensive Test Suite

Run the full test suite with 8 different tests:

```bash
python tests/test_ai_analysis.py
```

**Tests included:**
1. Module Imports - Verify all modules load correctly
2. Audio I/O - Test save/load functionality
3. Audio Editing - Test normalize, fade, gain functions
4. Analyzer Initialization - Test AI analyzer setup
5. Audio Analysis - Test real audio analysis
6. Full Pipeline - Test complete workflow
7. Configuration - Test API key and options
8. Error Handling - Test error cases

**Output Format:**
- ✓ Green checkmarks for passed tests
- ✗ Red X for failures
- ℹ Blue info for non-critical issues

## Running Individual Examples

Test specific functionality with the example scripts:

```bash
# View all examples
python examples/ai_analysis_examples.py

# Or run individual examples (uncomment in the file):
# - example_1_basic_analysis()
# - example_2_with_api_key()
# - example_3_speech_recognition()
# - example_4_with_audio_features()
# - example_5_different_analysis_types()
# - example_6_custom_model()
# - example_7_batch_processing()
# - example_8_integration_with_edit()
```

## First Time Setup

### 1. Install Dependencies

```bash
# Basic installation (audiopy_ai core)
pip install -e .

# With AI analysis support
pip install transformers torch librosa

# Optional: GPU support (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Quick Test

```bash
python quick_test.py
```

### 3. Run Full Suite

```bash
python tests/test_ai_analysis.py
```

## Testing with Your Own Audio Files

Create a simple test script:

```python
from audiopy_ai.ai_analysis import SoundComponentExtractor
from audiopy_ai.io import load

# Load your audio
samples, sr = load("your_audio.wav")

# Analyze
analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
results = analyzer.extract_components("your_audio.wav")

# View results
import json
print(json.dumps(results, indent=2, default=str))
```

## Troubleshooting

### Issue: "No module named 'transformers'"
**Solution:** Install transformers
```bash
pip install transformers torch
```

### Issue: "CUDA out of memory"
**Solution:** Use a smaller model
```python
from audiopy_ai.ai_analysis import AudioAnalyzer
analyzer = AudioAnalyzer(model_name="facebook/wav2vec2-base")
```

### Issue: Models downloading slowly
**Solution:** First download will be slow (~500MB). Subsequent runs will be instant.

### Issue: "Failed to initialize model"
**Solution:** Check your internet connection. Models need to download from HuggingFace.

## Test Results Interpretation

### ✓ Green (Passed)
Test completed successfully without errors.

### ✗ Red (Failed)
Test encountered an error. Check the error message and ensure dependencies are installed.

### ℹ️ Blue Info
Non-critical information or expected warnings (e.g., "Model downloading").

### ⚠️ Yellow Warning
Expected behavior (e.g., "Models need to download on first use").

## CI/CD Integration

Run tests automatically:

```bash
# Run all tests and return exit code
python tests/test_ai_analysis.py

# Run quick test
python quick_test.py
```

Both scripts return:
- Exit code `0` if all tests pass
- Exit code `1` if any test fails

## Performance Notes

- **First run:** 2-5 minutes (models download)
- **Subsequent runs:** A few seconds per audio file
- **GPU:** 10-50x faster with CUDA support

## Next Steps

After tests pass:
1. Read [AI_ANALYSIS_GUIDE.md](../AI_ANALYSIS_GUIDE.md) for detailed usage
2. Check [examples/ai_analysis_examples.py](../examples/ai_analysis_examples.py) for code samples
3. Integrate into your project using the API shown in examples
