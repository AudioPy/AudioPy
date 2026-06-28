"""
audiopy.utils — Utility Functions
===================================
Shared helper utilities for audio conversion, format detection, decibel
conversions, and audio validation used across the audiopy package.
"""

import numpy as np
import os

__all__ = [
    "db_to_amplitude",
    "amplitude_to_db",
    "is_valid_audio",
    "detect_format",
    "samples_to_time",
    "time_to_samples",
    "audio_stats",
    "ensure_mono",
    "ensure_float32",
    "frame_to_time",
    "time_to_frame",
]


def db_to_amplitude(db: float) -> float:
    """
    Convert a decibel value to a linear amplitude ratio.

    Parameters
    ----------
    db : float
        Decibel value. 0 dB = amplitude of 1.0, -6 dB ≈ 0.5, +6 dB ≈ 2.0.

    Returns
    -------
    float
        Linear amplitude ratio corresponding to the given dB value.

    Raises
    ------
    ValueError
        If db is not a finite number.

    Example
    -------
    >>> import audiopy as ap
    >>> amp = ap.utils.db_to_amplitude(-20.0)
    >>> print(f"{amp:.4f}")  # 0.1000
    """
    if not np.isfinite(db):
        raise ValueError(f"db must be a finite number, got {db}.")
    return float(10 ** (db / 20.0))


def amplitude_to_db(amplitude: float) -> float:
    """
    Convert a linear amplitude ratio to decibels.

    Parameters
    ----------
    amplitude : float
        Linear amplitude value. Must be non-negative.
        0.0 is treated as -inf dB.

    Returns
    -------
    float
        Decibel value. Returns ``float('-inf')`` if amplitude is 0.

    Raises
    ------
    ValueError
        If amplitude is negative.

    Example
    -------
    >>> import audiopy as ap
    >>> db = ap.utils.amplitude_to_db(0.5)
    >>> print(f"{db:.2f} dB")  # -6.02 dB
    """
    if amplitude < 0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}.")
    if amplitude == 0:
        return float("-inf")
    return float(20.0 * np.log10(amplitude))


def is_valid_audio(y: np.ndarray, sr: int) -> bool:
    """
    Check if a numpy array is a valid audio signal.

    Validates that ``y`` is a non-empty numpy array with finite values,
    and that ``sr`` is a positive integer.

    Parameters
    ----------
    y : np.ndarray
        Audio array to validate.
    sr : int
        Sample rate to validate.

    Returns
    -------
    bool
        ``True`` if the audio is valid, ``False`` otherwise.

    Example
    -------
    >>> import audiopy as ap
    >>> import numpy as np
    >>> y = np.zeros(22050, dtype=np.float32)
    >>> print(ap.utils.is_valid_audio(y, 22050))  # True
    >>> print(ap.utils.is_valid_audio(None, 22050))  # False
    """
    if not isinstance(y, np.ndarray):
        return False
    if y.size == 0:
        return False
    if not np.all(np.isfinite(y)):
        return False
    if not isinstance(sr, int) or sr <= 0:
        return False
    return True


def detect_format(path: str) -> str:
    """
    Detect the audio format of a file based on its file extension.

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    str
        Detected format string in uppercase, e.g., ``"WAV"``, ``"MP3"``,
        ``"FLAC"``, ``"OGG"``, ``"M4A"``, or ``"UNKNOWN"``.

    Example
    -------
    >>> import audiopy as ap
    >>> fmt = ap.utils.detect_format("audio.mp3")
    >>> print(fmt)  # "MP3"
    """
    _, ext = os.path.splitext(path)
    ext_map = {
        ".wav": "WAV",
        ".mp3": "MP3",
        ".flac": "FLAC",
        ".ogg": "OGG",
        ".m4a": "M4A",
        ".aiff": "AIFF",
        ".aif": "AIFF",
        ".opus": "OPUS",
        ".wma": "WMA",
    }
    return ext_map.get(ext.lower(), "UNKNOWN")


def samples_to_time(n_samples: int, sr: int) -> float:
    """
    Convert a number of audio samples to time in seconds.

    Parameters
    ----------
    n_samples : int
        Number of audio samples.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    float
        Time in seconds.

    Raises
    ------
    ValueError
        If n_samples is negative or sr is not positive.

    Example
    -------
    >>> import audiopy as ap
    >>> t = ap.utils.samples_to_time(44100, 44100)
    >>> print(t)  # 1.0
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}.")
    if sr <= 0:
        raise ValueError(f"sr must be a positive integer, got {sr}.")
    return float(n_samples) / float(sr)


def time_to_samples(time_s: float, sr: int) -> int:
    """
    Convert a time value in seconds to the corresponding sample index.

    Parameters
    ----------
    time_s : float
        Time in seconds. Must be non-negative.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    int
        Corresponding sample index (rounded down).

    Raises
    ------
    ValueError
        If time_s is negative or sr is not positive.

    Example
    -------
    >>> import audiopy as ap
    >>> n = ap.utils.time_to_samples(2.5, 44100)
    >>> print(n)  # 110250
    """
    if time_s < 0:
        raise ValueError(f"time_s must be non-negative, got {time_s}.")
    if sr <= 0:
        raise ValueError(f"sr must be a positive integer, got {sr}.")
    return int(time_s * sr)


def audio_stats(y: np.ndarray, sr: int) -> dict:
    """
    Compute summary statistics of an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"duration_s"``    : float, duration in seconds
        - ``"n_samples"``     : int, total number of samples
        - ``"sample_rate"``   : int, sample rate in Hz
        - ``"channels"``      : int, number of channels (1 or 2)
        - ``"peak_amplitude"``: float, maximum absolute amplitude
        - ``"rms"``           : float, root-mean-square amplitude
        - ``"peak_db"``       : float, peak amplitude in dBFS
        - ``"rms_db"``        : float, RMS amplitude in dBFS
        - ``"dtype"``         : str, numpy dtype of the array

    Raises
    ------
    ValueError
        If y is not a valid numpy array.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> stats = ap.utils.audio_stats(y, sr)
    >>> print(stats)
    """
    if not isinstance(y, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(y).__name__}.")

    n_samples = y.shape[-1] if y.ndim == 2 else y.shape[0]
    channels = y.shape[0] if y.ndim == 2 else 1
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))

    return {
        "duration_s": float(n_samples) / float(sr),
        "n_samples": int(n_samples),
        "sample_rate": int(sr),
        "channels": int(channels),
        "peak_amplitude": peak,
        "rms": rms,
        "peak_db": amplitude_to_db(peak) if peak > 0 else float("-inf"),
        "rms_db": amplitude_to_db(rms) if rms > 0 else float("-inf"),
        "dtype": str(y.dtype),
    }


def ensure_mono(y: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono by averaging channels if stereo.

    Parameters
    ----------
    y : np.ndarray
        Audio array. Shape: (samples,) or (channels, samples).

    Returns
    -------
    np.ndarray
        Mono audio as float32 1-D array.

    Example
    -------
    >>> import audiopy as ap
    >>> import numpy as np
    >>> y_stereo = np.random.randn(2, 44100).astype(np.float32)
    >>> y_mono = ap.utils.ensure_mono(y_stereo)
    >>> print(y_mono.shape)  # (44100,)
    """
    if y.ndim == 2:
        return np.mean(y, axis=0).astype(np.float32)
    return y.astype(np.float32)


def ensure_float32(y: np.ndarray) -> np.ndarray:
    """
    Convert an audio array to float32 dtype.

    Also handles integer PCM formats (int16, int32) by normalizing to [-1, 1].

    Parameters
    ----------
    y : np.ndarray
        Audio array of any numeric dtype.

    Returns
    -------
    np.ndarray
        Audio as float32, normalized to [-1, 1] if integer input.

    Example
    -------
    >>> import audiopy as ap
    >>> import numpy as np
    >>> y_int16 = np.array([0, 16384, -16384], dtype=np.int16)
    >>> y_float = ap.utils.ensure_float32(y_int16)
    >>> print(y_float)  # [0.  0.5 -0.5]
    """
    if y.dtype == np.float32:
        return y
    if y.dtype == np.float64:
        return y.astype(np.float32)
    if y.dtype == np.int16:
        return (y.astype(np.float32) / 32768.0)
    if y.dtype == np.int32:
        return (y.astype(np.float32) / 2147483648.0)
    if y.dtype == np.uint8:
        return (y.astype(np.float32) - 128.0) / 128.0
    return y.astype(np.float32)


def frame_to_time(frame: int, sr: int, hop_length: int = 512) -> float:
    """
    Convert a frame index to time in seconds.

    Parameters
    ----------
    frame : int
        Frame index (0-based).
    sr : int
        Sample rate in Hz.
    hop_length : int, optional
        Number of samples between consecutive frames. Default is 512.

    Returns
    -------
    float
        Time in seconds corresponding to the given frame index.

    Example
    -------
    >>> import audiopy as ap
    >>> t = ap.utils.frame_to_time(100, sr=22050, hop_length=512)
    >>> print(f"{t:.3f} s")
    """
    return float(frame * hop_length) / float(sr)


def time_to_frame(time_s: float, sr: int, hop_length: int = 512) -> int:
    """
    Convert a time in seconds to the nearest frame index.

    Parameters
    ----------
    time_s : float
        Time in seconds.
    sr : int
        Sample rate in Hz.
    hop_length : int, optional
        Number of samples between consecutive frames. Default is 512.

    Returns
    -------
    int
        Nearest frame index corresponding to the given time.

    Example
    -------
    >>> import audiopy as ap
    >>> frame = ap.utils.time_to_frame(1.0, sr=22050, hop_length=512)
    >>> print(frame)
    """
    return int(time_s * sr / hop_length)
