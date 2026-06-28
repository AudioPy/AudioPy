"""
audiopy.features — Audio Feature Extraction Module
====================================================
Spectral, temporal, and rhythmic feature extraction using librosa.
All features returned as float32 numpy arrays.
"""

import numpy as np

__all__ = [
    "mfcc",
    "mel_spectrogram",
    "spectrogram",
    "chroma",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "rms_energy",
    "tempo",
    "pitch",
    "beat_frames",
    "extract_all",
]


def _ensure_mono(y: np.ndarray) -> np.ndarray:
    """Internal helper: convert to mono float32 if needed."""
    if y.ndim == 2:
        return np.mean(y, axis=0).astype(np.float32)
    return y.astype(np.float32)


def mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs).

    MFCCs are widely used features for speech and audio recognition tasks.
    They compactly represent the spectral envelope of an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    n_mfcc : int, optional
        Number of MFCCs to compute. Default is 13.

    Returns
    -------
    np.ndarray
        MFCC matrix of shape (n_mfcc, time_frames), dtype float32.

    Raises
    ------
    ValueError
        If n_mfcc is not a positive integer.
    RuntimeError
        If MFCC computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("speech.wav")
    >>> mfccs = ap.mfcc(y, sr, n_mfcc=13)
    >>> print(mfccs.shape)  # (13, T)
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    if n_mfcc <= 0:
        raise ValueError(f"n_mfcc must be a positive integer, got {n_mfcc}.")

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=n_mfcc)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"MFCC computation failed.\nOriginal error: {e}") from e


def mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    """
    Compute a Mel-scale power spectrogram.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    n_mels : int, optional
        Number of Mel frequency bands. Default is 128.

    Returns
    -------
    np.ndarray
        Mel spectrogram of shape (n_mels, time_frames), dtype float32.
        Values are in power (amplitude squared), not dB.

    Raises
    ------
    ValueError
        If n_mels is not a positive integer.
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("music.wav")
    >>> mel_spec = ap.mel_spectrogram(y, sr, n_mels=128)
    >>> print(mel_spec.shape)  # (128, T)
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    if n_mels <= 0:
        raise ValueError(f"n_mels must be a positive integer, got {n_mels}.")

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=n_mels)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Mel spectrogram computation failed.\nOriginal error: {e}") from e


def spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute the magnitude spectrogram using the Short-Time Fourier Transform (STFT).

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Magnitude spectrogram of shape (1 + n_fft/2, time_frames), dtype float32.

    Raises
    ------
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> S = ap.spectrogram(y, sr)
    >>> print(S.shape)  # (1025, T) for default n_fft=2048
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        stft_matrix = librosa.stft(y_mono)
        result = np.abs(stft_matrix)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Spectrogram computation failed.\nOriginal error: {e}") from e


def chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a chromagram from an audio signal.

    The chromagram represents the intensity of each of the 12 pitch classes
    (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) over time.
    Useful for chord recognition and music analysis.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Chroma features of shape (12, time_frames), dtype float32.

    Raises
    ------
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("song.wav")
    >>> chroma_feat = ap.chroma(y, sr)
    >>> print(chroma_feat.shape)  # (12, T)
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.chroma_stft(y=y_mono, sr=sr)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Chroma computation failed.\nOriginal error: {e}") from e


def zero_crossing_rate(y: np.ndarray) -> np.ndarray:
    """
    Compute the zero-crossing rate of an audio signal.

    The zero-crossing rate is the rate at which the signal changes sign.
    Useful for detecting percussive sounds and voiced/unvoiced speech.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.

    Returns
    -------
    np.ndarray
        Zero-crossing rate of shape (1, time_frames), dtype float32.

    Raises
    ------
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> zcr = ap.zero_crossing_rate(y)
    >>> print(f"Mean ZCR: {zcr.mean():.4f}")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.zero_crossing_rate(y_mono)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Zero-crossing rate computation failed.\nOriginal error: {e}") from e


def spectral_centroid(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute the spectral centroid of an audio signal.

    The spectral centroid indicates where the "centre of mass" of the
    spectrum is located. A higher centroid corresponds to a "brighter" sound.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Spectral centroid frequencies of shape (1, time_frames), dtype float32,
        in Hz.

    Raises
    ------
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> sc = ap.spectral_centroid(y, sr)
    >>> print(f"Mean spectral centroid: {sc.mean():.1f} Hz")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Spectral centroid computation failed.\nOriginal error: {e}") from e


def spectral_bandwidth(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute the spectral bandwidth of an audio signal.

    Measures the width of the spectral band around the centroid.
    Indicates how "wide" or "narrow" the spectrum is.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Spectral bandwidth of shape (1, time_frames), dtype float32, in Hz.

    Raises
    ------
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> bw = ap.spectral_bandwidth(y, sr)
    >>> print(f"Mean bandwidth: {bw.mean():.1f} Hz")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Spectral bandwidth computation failed.\nOriginal error: {e}") from e


def spectral_rolloff(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute the spectral roll-off frequency of an audio signal.

    The roll-off frequency is the frequency below which a specified percentage
    (default 85%) of the total spectral energy lies. Used to distinguish
    between harmonic and percussive content.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Spectral roll-off frequency of shape (1, time_frames), dtype float32,
        in Hz.

    Raises
    ------
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> rolloff = ap.spectral_rolloff(y, sr)
    >>> print(f"Mean rolloff: {rolloff.mean():.1f} Hz")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Spectral rolloff computation failed.\nOriginal error: {e}") from e


def rms_energy(y: np.ndarray) -> np.ndarray:
    """
    Compute the Root Mean Square (RMS) energy of an audio signal.

    RMS energy represents the loudness/power of the audio over time.
    Useful for voice activity detection and dynamic range analysis.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.

    Returns
    -------
    np.ndarray
        RMS energy of shape (1, time_frames), dtype float32.

    Raises
    ------
    RuntimeError
        If computation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> rms = ap.rms_energy(y)
    >>> print(f"Mean RMS energy: {rms.mean():.4f}")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        result = librosa.feature.rms(y=y_mono)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"RMS energy computation failed.\nOriginal error: {e}") from e


def tempo(y: np.ndarray, sr: int) -> float:
    """
    Estimate the tempo (BPM) of an audio signal.

    Uses librosa's beat tracker to estimate the global tempo.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    float
        Estimated tempo in beats per minute (BPM).

    Raises
    ------
    RuntimeError
        If tempo estimation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("music.wav")
    >>> bpm = ap.tempo(y, sr)
    >>> print(f"Estimated tempo: {bpm:.1f} BPM")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        tempo_val, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
        # Handle array return in newer librosa versions
        if hasattr(tempo_val, '__len__'):
            return float(tempo_val[0])
        return float(tempo_val)
    except Exception as e:
        raise RuntimeError(f"Tempo estimation failed.\nOriginal error: {e}") from e


def pitch(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate the fundamental frequency (pitch) over time using pYIN algorithm.

    Uses librosa's ``pyin`` function, which is a probabilistic variant of the
    YIN algorithm and is suitable for monophonic pitch detection.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Array of estimated fundamental frequencies in Hz per frame.
        Frames where pitch is undetected contain ``np.nan``.
        Shape: (time_frames,), dtype float32.

    Raises
    ------
    RuntimeError
        If pitch estimation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("melody.wav")
    >>> f0 = ap.pitch(y, sr)
    >>> print(f"Detected pitch frames: {(~np.isnan(f0)).sum()}")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_mono,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        return f0.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Pitch estimation failed.\nOriginal error: {e}") from e


def beat_frames(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect beat frame indices in an audio signal.

    Uses librosa's beat tracker to find frame positions of detected beats.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Array of beat frame indices as integers.
        Convert to timestamps: ``librosa.frames_to_time(beat_frames, sr=sr)``

    Raises
    ------
    RuntimeError
        If beat detection fails.

    Example
    -------
    >>> import audiopy as ap
    >>> import librosa
    >>> y, sr = ap.load("music.wav")
    >>> beats = ap.beat_frames(y, sr)
    >>> times = librosa.frames_to_time(beats, sr=sr)
    >>> print(f"Detected {len(beats)} beats")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = _ensure_mono(y)
    try:
        _, beats = librosa.beat.beat_track(y=y_mono, sr=sr)
        return beats.astype(np.int32)
    except Exception as e:
        raise RuntimeError(f"Beat detection failed.\nOriginal error: {e}") from e


def extract_all(y: np.ndarray, sr: int) -> dict:
    """
    Extract all available audio features in a single call.

    Computes all features and returns them in a dictionary. Useful for
    building feature matrices for machine learning pipelines.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    dict
        Dictionary with the following keys (all values are float32 np.ndarray
        unless noted):

        - ``"mfcc"``                : shape (13, T)
        - ``"mel_spectrogram"``     : shape (128, T)
        - ``"spectrogram"``         : shape (F, T)
        - ``"chroma"``              : shape (12, T)
        - ``"zero_crossing_rate"``  : shape (1, T)
        - ``"spectral_centroid"``   : shape (1, T)
        - ``"spectral_bandwidth"``  : shape (1, T)
        - ``"spectral_rolloff"``    : shape (1, T)
        - ``"rms_energy"``          : shape (1, T)
        - ``"tempo"``               : float (scalar BPM)
        - ``"pitch"``               : shape (T,)
        - ``"beat_frames"``         : shape (N,) — integer frame indices

    Raises
    ------
    RuntimeError
        If any feature extraction fails. Other features are still returned.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> features = ap.extract_all(y, sr)
    >>> for name, val in features.items():
    ...     print(f"{name}: {val.shape if hasattr(val, 'shape') else val}")
    """
    features = {}
    errors = []

    extractors = {
        "mfcc": lambda: mfcc(y, sr),
        "mel_spectrogram": lambda: mel_spectrogram(y, sr),
        "spectrogram": lambda: spectrogram(y, sr),
        "chroma": lambda: chroma(y, sr),
        "zero_crossing_rate": lambda: zero_crossing_rate(y),
        "spectral_centroid": lambda: spectral_centroid(y, sr),
        "spectral_bandwidth": lambda: spectral_bandwidth(y, sr),
        "spectral_rolloff": lambda: spectral_rolloff(y, sr),
        "rms_energy": lambda: rms_energy(y),
        "tempo": lambda: tempo(y, sr),
        "pitch": lambda: pitch(y, sr),
        "beat_frames": lambda: beat_frames(y, sr),
    }

    for name, extractor in extractors.items():
        try:
            features[name] = extractor()
        except Exception as e:
            features[name] = None
            errors.append(f"  - {name}: {e}")

    if errors:
        import warnings
        warnings.warn(
            "Some features could not be extracted:\n" + "\n".join(errors),
            RuntimeWarning,
            stacklevel=2,
        )

    return features
