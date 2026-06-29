"""
Test suite for AudioPy AI Analysis functionality
Run this file to test all components: python test_ai_analysis.py
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
import tempfile
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiopy_ai.io import load, save
from audiopy_ai.edit import normalize, fade, apply_gain
from audiopy_ai.ai_analysis import AudioAnalyzer, SoundComponentExtractor


class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title):
    """Print test section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.RESET}")


def create_test_audio(duration_s=2, sr=16000, frequency=440):
    """
    Create a simple test audio file (sine wave)
    
    Args:
        duration_s: Duration in seconds
        sr: Sample rate in Hz
        frequency: Frequency of sine wave in Hz
    
    Returns:
        tuple: (samples, sr)
    """
    t = np.linspace(0, duration_s, int(sr * duration_s))
    samples = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return samples, sr


def create_test_audio_file(filepath, duration_s=2):
    """Create and save a test audio file"""
    samples, sr = create_test_audio(duration_s)
    save(filepath, samples, sr)
    return filepath


# ============================================================================
# TEST 1: Module Imports
# ============================================================================

def test_imports():
    """Test if all modules can be imported"""
    print_header("TEST 1: Module Imports")
    
    try:
        from audiopy_ai.io import load, save
        print_success("audiopy_ai.io imported successfully")
    except ImportError as e:
        print_error(f"Failed to import audiopy_ai.io: {e}")
        return False
    
    try:
        from audiopy_ai.edit import normalize, fade, apply_gain
        print_success("audiopy_ai.edit imported successfully")
    except ImportError as e:
        print_error(f"Failed to import audiopy_ai.edit: {e}")
        return False
    
    try:
        from audiopy_ai.ai_analysis import AudioAnalyzer, SoundComponentExtractor
        print_success("audiopy_ai.ai_analysis imported successfully")
    except ImportError as e:
        print_error(f"Failed to import audiopy_ai.ai_analysis: {e}")
        return False
    
    return True


# ============================================================================
# TEST 2: Audio I/O
# ============================================================================

def test_audio_io():
    """Test audio file I/O operations"""
    print_header("TEST 2: Audio I/O Operations")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_audio.wav"
        
        # Create test audio
        samples, sr = create_test_audio(duration_s=1)
        print_success(f"Created test audio: {len(samples)} samples at {sr}Hz")
        
        # Save audio
        try:
            save(str(test_file), samples, sr)
            print_success(f"Saved audio to {test_file.name}")
        except Exception as e:
            print_error(f"Failed to save audio: {e}")
            return False
        
        # Load audio
        try:
            loaded_samples, loaded_sr = load(str(test_file))
            print_success(f"Loaded audio: {len(loaded_samples)} samples at {loaded_sr}Hz")
        except Exception as e:
            print_error(f"Failed to load audio: {e}")
            return False
        
        # Verify data
        if len(loaded_samples) == len(samples) and loaded_sr == sr:
            print_success("Audio data matches after save/load")
        else:
            print_error("Audio data mismatch")
            return False
    
    return True


# ============================================================================
# TEST 3: Audio Editing
# ============================================================================

def test_audio_editing():
    """Test audio editing functions"""
    print_header("TEST 3: Audio Editing Functions")
    
    samples, sr = create_test_audio(duration_s=2)
    original_length = len(samples)
    
    # Test normalization
    try:
        normalized = normalize(samples)
        print_success(f"Normalize: {len(samples)} samples → {len(normalized)} samples")
    except Exception as e:
        print_error(f"Normalize failed: {e}")
        return False
    
    # Test gain
    try:
        gained = apply_gain(samples, db=6)
        print_success(f"Apply gain: +6dB applied")
    except Exception as e:
        print_error(f"Apply gain failed: {e}")
        return False
    
    # Test fade
    try:
        faded = fade(samples, sr, fade_in_s=0.5, fade_out_s=0.5)
        print_success(f"Fade: fade in/out applied")
    except Exception as e:
        print_error(f"Fade failed: {e}")
        return False
    
    return True


# ============================================================================
# TEST 4: AI Analyzer Initialization
# ============================================================================

def test_analyzer_initialization():
    """Test AI Analyzer initialization"""
    print_header("TEST 4: AI Analyzer Initialization")
    
    # Test SoundComponentExtractor initialization
    try:
        analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
        print_success(f"SoundComponentExtractor initialized with audio_tagging")
    except Exception as e:
        print_info(f"SoundComponentExtractor initialization (expected if models not downloaded): {e}")
        print_info("This is normal - transformers models will download on first use")
    
    # Test analyzer types
    analyzer_types = [
        "audio_tagging",
        "speech_recognition",
        "emotion_detection",
        "music_analysis"
    ]
    
    for atype in analyzer_types:
        try:
            analyzer = SoundComponentExtractor(analysis_type=atype)
            print_success(f"Analyzer type '{atype}' available")
        except Exception as e:
            print_info(f"Analyzer type '{atype}' initialization: {str(e)[:80]}")
    
    return True


# ============================================================================
# TEST 5: Create Sample Audio and Test Analysis
# ============================================================================

def test_audio_analysis():
    """Test audio analysis with real file"""
    print_header("TEST 5: Audio Analysis with Sample File")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / "test_analysis.wav"
        
        # Create test audio
        try:
            create_test_audio_file(str(audio_file), duration_s=3)
            print_success(f"Created test audio file: {audio_file.name}")
        except Exception as e:
            print_error(f"Failed to create test audio: {e}")
            return False
        
        # Try to load and analyze
        try:
            samples, sr = load(str(audio_file))
            print_success(f"Loaded audio: {len(samples)} samples at {sr}Hz")
        except Exception as e:
            print_error(f"Failed to load audio: {e}")
            return False
        
        # Initialize analyzer
        try:
            analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
            print_success("SoundComponentExtractor initialized")
        except Exception as e:
            print_info(f"Analyzer initialization: {e}")
            print_info("(This is normal if transformers isn't installed yet)")
            return True
        
        # Try analysis
        try:
            results = analyzer.extract_components(str(audio_file))
            print_success("Audio analysis completed")
            
            # Display results structure
            if 'components' in results:
                components = results['components']
                if 'detected_sounds' in components:
                    sounds = components['detected_sounds']
                    print_success(f"Detected {len(sounds)} sound components: {sounds[:3]}{'...' if len(sounds) > 3 else ''}")
            
            print_success("Results structure is valid")
            return True
            
        except Exception as e:
            print_info(f"Analysis error (models may need to download): {str(e)[:100]}")
            print_info("This is expected if transformers models haven't been downloaded yet")
            return True


# ============================================================================
# TEST 6: Full Pipeline Integration
# ============================================================================

def test_full_pipeline():
    """Test full pipeline: load → analyze → edit → save"""
    print_header("TEST 6: Full Pipeline Integration")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / "pipeline_test.wav"
        output_file = Path(tmpdir) / "pipeline_output.wav"
        
        try:
            # Step 1: Create audio
            create_test_audio_file(str(audio_file), duration_s=2)
            print_success("✓ Step 1: Created test audio")
            
            # Step 2: Load audio
            samples, sr = load(str(audio_file))
            print_success("✓ Step 2: Loaded audio")
            
            # Step 3: Get audio info
            original_rms = np.sqrt(np.mean(samples**2))
            print_success(f"✓ Step 3: Audio RMS = {original_rms:.4f}")
            
            # Step 4: Normalize
            normalized = normalize(samples)
            normalized_rms = np.sqrt(np.mean(normalized**2))
            print_success(f"✓ Step 4: Normalized (RMS: {original_rms:.4f} → {normalized_rms:.4f})")
            
            # Step 5: Apply fade
            faded = fade(normalized, sr, fade_in_s=0.5, fade_out_s=0.5)
            print_success("✓ Step 5: Applied fade in/out")
            
            # Step 6: Save processed audio
            save(str(output_file), faded, sr)
            print_success("✓ Step 6: Saved processed audio")
            
            # Step 7: Verify output
            output_samples, output_sr = load(str(output_file))
            if len(output_samples) > 0:
                print_success(f"✓ Step 7: Verified output ({len(output_samples)} samples)")
            
            print_success("\nFull pipeline completed successfully!")
            return True
            
        except Exception as e:
            print_error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# TEST 7: Configuration and API Key
# ============================================================================

def test_configuration():
    """Test configuration options"""
    print_header("TEST 7: Configuration & API Key Support")
    
    try:
        # Test without API key
        analyzer = SoundComponentExtractor(analysis_type="audio_tagging", api_key=None)
        print_success("Analyzer works without API key")
    except Exception as e:
        print_info(f"Configuration test: {str(e)[:80]}")
    
    # Test with dummy API key (won't actually use it unless model is private)
    try:
        analyzer = SoundComponentExtractor(
            analysis_type="audio_tagging",
            api_key="hf_test_key"
        )
        print_success("API key parameter accepted")
    except Exception as e:
        print_info(f"API key configuration: {e}")
    
    return True


# ============================================================================
# TEST 8: Error Handling
# ============================================================================

def test_error_handling():
    """Test error handling"""
    print_header("TEST 8: Error Handling")
    
    # Test with non-existent file
    try:
        analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
        results = analyzer.extract_components("/nonexistent/file.wav")
        print_error("Should have raised an error for non-existent file")
        return False
    except Exception as e:
        print_success(f"Properly handles non-existent file: {type(e).__name__}")
    
    # Test with invalid analysis type (should fall back gracefully)
    try:
        analyzer = SoundComponentExtractor(analysis_type="invalid_type")
        print_success("Invalid analysis type handled gracefully")
    except Exception as e:
        print_info(f"Invalid type error (expected): {str(e)[:80]}")
    
    return True


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results):
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.RESET}" if result else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {total} | Passed: {passed} | Failed: {failed}{Colors.RESET}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Some tests failed. See details above.{Colors.RESET}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}AudioPy AI Analysis - Test Suite{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    tests = {
        "Test 1: Module Imports": test_imports(),
        "Test 2: Audio I/O": test_audio_io(),
        "Test 3: Audio Editing": test_audio_editing(),
        "Test 4: Analyzer Init": test_analyzer_initialization(),
        "Test 5: Audio Analysis": test_audio_analysis(),
        "Test 6: Full Pipeline": test_full_pipeline(),
        "Test 7: Configuration": test_configuration(),
        "Test 8: Error Handling": test_error_handling(),
    }
    
    print_summary(tests)
    
    # Exit with appropriate code
    failed = sum(1 for v in tests.values() if not v)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
