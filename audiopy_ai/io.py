# audiopy_ai/io.py
import soundfile as sf
import numpy as np

def load(path, mono=True):
    data, sr = sf.read(path)
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype('float32'), sr

def save(path, samples, sr):
    sf.write(path, samples, sr)
