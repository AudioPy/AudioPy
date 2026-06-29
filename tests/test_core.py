"""
Unit Tests for AudioPy - Core Functionality (No AI Models Required)
Run this for quick validation: python tests/test_core.py
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiopy_ai.io import load, save
from audiopy_ai.edit import (
    trim, concat, apply_gain, rms_db, normalize, fade, reverse
)


class TestAudioIO(unittest.TestCase):
    """Test audio file I/O operations"""
    
    def setUp(self):
        """Create test audio"""
        self.sr = 16000
        self.duration = 1
        t = np.linspace(0, self.duration, self.sr * self.duration)
        self.samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    def test_save_and_load(self):
        """Test save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.wav"
            
            # Save
            save(str(filepath), self.samples, self.sr)
            self.assertTrue(filepath.exists(), "File not saved")
            
            # Load
            loaded_samples, loaded_sr = load(str(filepath))
            
            self.assertEqual(len(loaded_samples), len(self.samples))
            self.assertEqual(loaded_sr, self.sr)
    
    def test_load_mono(self):
        """Test loading as mono"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.wav"
            save(str(filepath), self.samples, self.sr)
            
            loaded_samples, _ = load(str(filepath), mono=True)
            self.assertEqual(loaded_samples.ndim, 1, "Should be mono")


class TestAudioEditing(unittest.TestCase):
    """Test audio editing functions"""
    
    def setUp(self):
        """Create test audio"""
        self.sr = 16000
        self.duration = 2
        t = np.linspace(0, self.duration, self.sr * self.duration)
        self.samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    def test_trim(self):
        """Test trim function"""
        trimmed = trim(self.samples, self.sr, 0.5, 1.5)
        expected_len = int((1.5 - 0.5) * self.sr)
        self.assertEqual(len(trimmed), expected_len)
    
    def test_concat(self):
        """Test concat function"""
        concat_samples = concat([self.samples, self.samples])
        self.assertEqual(len(concat_samples), len(self.samples) * 2)
    
    def test_apply_gain(self):
        """Test gain application"""
        gained = apply_gain(self.samples, db=6)
        
        # Gained version should be louder
        self.assertGreater(
            np.abs(gained).max(),
            np.abs(self.samples).max()
        )
    
    def test_rms_db(self):
        """Test RMS calculation"""
        rms = rms_db(self.samples)
        self.assertIsInstance(rms, (float, np.floating))
        self.assertLess(rms, 0)  # Should be negative dB
    
    def test_normalize(self):
        """Test normalization"""
        normalized = normalize(self.samples, target_db=-20)
        
        # Normalized should match target RMS
        actual_rms = rms_db(normalized)
        self.assertAlmostEqual(actual_rms, -20, delta=0.1)
    
    def test_fade(self):
        """Test fade in/out"""
        faded = fade(self.samples, self.sr, fade_in_s=0.5, fade_out_s=0.5)
        
        # Same length
        self.assertEqual(len(faded), len(self.samples))
        
        # Start should be low (fade in)
        self.assertLess(np.abs(faded[10]), np.abs(self.samples[10]))
        
        # End should be low (fade out)
        self.assertLess(np.abs(faded[-10]), np.abs(self.samples[-10]))
    
    def test_reverse(self):
        """Test reverse function"""
        reversed_samples = reverse(self.samples)
        
        # Should be reversed
        np.testing.assert_array_almost_equal(
            reversed_samples,
            self.samples[::-1]
        )


class TestAIAnalysis(unittest.TestCase):
    """Test AI analysis module initialization"""
    
    def test_imports(self):
        """Test that AI analysis modules can be imported"""
        try:
            from audiopy_ai.ai_analysis import AudioAnalyzer, SoundComponentExtractor
            self.assertTrue(True, "Import successful")
        except ImportError as e:
            self.fail(f"Failed to import AI analysis: {e}")
    
    def test_analyzer_class_exists(self):
        """Test that analyzer classes exist"""
        from audiopy_ai.ai_analysis import AudioAnalyzer, SoundComponentExtractor
        self.assertTrue(hasattr(AudioAnalyzer, '__init__'))
        self.assertTrue(hasattr(SoundComponentExtractor, '__init__'))


class TestIntegration(unittest.TestCase):
    """Test integration of multiple functions"""
    
    def setUp(self):
        """Create test audio"""
        self.sr = 16000
        self.duration = 1
        t = np.linspace(0, self.duration, self.sr * self.duration)
        self.samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    def test_pipeline(self):
        """Test complete pipeline: load → process → save"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.wav"
            output_file = Path(tmpdir) / "output.wav"
            
            # Save original
            save(str(input_file), self.samples, self.sr)
            
            # Load and process
            loaded, sr = load(str(input_file))
            processed = normalize(loaded)
            processed = fade(processed, sr, fade_in_s=0.2, fade_out_s=0.2)
            
            # Save processed
            save(str(output_file), processed, sr)
            
            # Verify
            self.assertTrue(output_file.exists())
            final_samples, final_sr = load(str(output_file))
            self.assertEqual(len(final_samples), len(processed))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAudioIO))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioEditing))
    suite.addTests(loader.loadTestsFromTestCase(TestAIAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
