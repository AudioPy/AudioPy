# audiopy_ai/effects/reverb.py
import numpy as np
from scipy.signal import fftconvolve

def reverb(samples, ir, wet=0.3):
    # normalize IR energy
    ir = ir / (np.linalg.norm(ir) + 1e-9)
    convolved = fftconvolve(samples, ir)[:len(samples)]
    return (1.0 - wet) * samples + wet * convolved
