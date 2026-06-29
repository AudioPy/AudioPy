# audiopy_ai/edit.py
import numpy as np

def trim(samples, sr, start_s, end_s):
    start = int(start_s * sr)
    end = int(end_s * sr)
    return samples[start:end]

def concat(list_of_samples):
    return np.concatenate(list_of_samples)

def apply_gain(samples, db):
    factor = 10 ** (db / 20.0)
    return samples * factor

def rms_db(samples, eps=1e-9):
    rms = np.sqrt(np.mean(samples**2) + eps)
    return 20 * np.log10(rms)

def normalize(samples, target_db=-20.0):
    cur_db = rms_db(samples)
    gain_db = target_db - cur_db
    return apply_gain(samples, gain_db)

def fade(samples, sr, fade_in_s=0.5, fade_out_s=0.5):
    n = len(samples)
    # fade in
    fi = int(fade_in_s * sr)
    fo = int(fade_out_s * sr)
    env = np.ones(n)
    if fi > 0:
        env[:fi] = np.linspace(0,1,fi)
    if fo > 0:
        env[-fo:] = np.linspace(1,0,fo)
    return samples * env

def reverse(samples):
    return samples[::-1]
