# tests/test_io.py
import numpy as np
import soundfile as sf
import tempfile
import os

from audiopy_ai.io import load, save


def generate_sine_wave(freq=440, sr=22050, duration=1.0, amplitude=0.5):
    """Generate a simple sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def test_save_and_load_roundtrip():
    """Verify that saving and reloading a WAV file preserves shape and values."""
    samples = generate_sine_wave()
    sr = 22050

    # create a temporary WAV file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.wav")

        # 1. Save using our library
        save(path, samples, sr)

        # 2. Check file exists
        assert os.path.exists(path), "File was not created"

        # 3. Load it back using our library
        loaded, sr_loaded = load(path)

        # 4. Verify sampling rate and shape
        assert sr_loaded == sr, "Sample rate mismatch"
        assert loaded.shape == samples.shape, "Loaded shape mismatch"

        # 5. Verify that waveform is almost identical (allowing small numeric diff)
        assert np.allclose(samples, loaded, rtol=1e-5, atol=1e-4), "Waveform mismatch"


def test_load_mono_conversion():
    """Ensure stereo input is converted to mono when mono=True."""
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)], axis=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stereo.wav")
        sf.write(path, stereo, sr)

        mono, sr_loaded = load(path, mono=True)

        assert sr_loaded == sr
        assert mono.ndim == 1, "Mono conversion failed"
        assert abs(np.mean(mono) - 0) < 0.1, "Unexpected mean after mono mix"
