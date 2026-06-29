# audiopy_ai/effects/filters.py
from scipy.signal import butter, sosfilt

def butter_lowpass(sr, cutoff_hz, order=4):
    sos = butter(order, cutoff_hz, btype='low', fs=sr, output='sos')
    return sos

def butter_highpass(sr, cutoff_hz, order=4):
    sos = butter(order, cutoff_hz, btype='high', fs=sr, output='sos')
    return sos

def apply_sos_filter(samples, sos):
    return sosfilt(sos, samples)
