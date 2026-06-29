# tests/test_edit.py
import numpy as np
from audiopy_ai.edit import trim, normalize

def test_trim_length():
    sr = 22050
    samples = np.arange(0, sr * 5)  # 5 seconds dummy data
    result = trim(samples, sr, 1, 3)
    assert len(result) == sr * 2

def test_normalize_rms_change():
    sr = 22050
    samples = np.ones(sr) * 0.01
    norm = normalize(samples, -20.0)
    rms_old = np.sqrt(np.mean(samples ** 2))
    rms_new = np.sqrt(np.mean(norm ** 2))
    assert rms_new > rms_old
