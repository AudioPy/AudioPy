# audiopy_ai/effects/time.py
from scipy.signal import resample

def time_stretch(samples, rate):
    # rate >1 speeds up (shorter), <1 slows down (longer)
    n_new = int(len(samples) / rate)
    return resample(samples, n_new)
