"""
audiopy.io — Audio Input/Output Module
=======================================
Provides functions for loading, saving, recording, and inspecting audio files.
Supports mp3, wav, flac, ogg, m4a and more via librosa and soundfile backends.
"""

import os
import tempfile
import numpy as np

__all__ = [
    "load",
    "save",
    "record",
    "from_url",
    "info",
]


def load(path: str, sr: int = 22050):
    """
    Load an audio file from disk into a float32 numpy array.

    Supports: wav, mp3, flac, ogg, m4a, aiff, and any format supported by
    librosa (which uses soundfile + audioread as fallback).

    Parameters
    ----------
    path : str
        Absolute or relative path to the audio file.
    sr : int, optional
        Target sample rate to resample to. Default is 22050 Hz.
        Pass ``None`` to preserve the native sample rate.

    Returns
    -------
    y : np.ndarray
        Audio time-series as a float32 1-D (mono) or 2-D (stereo) array.
    sr : int
        Sample rate of the returned audio.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    RuntimeError
        If the file cannot be decoded or loaded.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("speech.wav")
    >>> print(f"Loaded {len(y)/sr:.2f} seconds at {sr} Hz")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required for audio loading. "
            "Install it with: pip install librosa"
        ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Audio file not found: '{path}'. "
            "Please check the path and try again."
        )

    try:
        y, sr_out = librosa.load(path, sr=sr, mono=False)
        # Ensure float32
        y = y.astype(np.float32)
        return y, int(sr_out)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load audio from '{path}'. "
            f"Make sure the file is a valid audio format.\nOriginal error: {e}"
        ) from e


def save(path: str, y: np.ndarray, sr: int, format: str = "wav") -> None:
    """
    Save a numpy audio array to a file on disk.

    Parameters
    ----------
    path : str
        Destination file path (including filename and extension).
    y : np.ndarray
        Audio time-series as a float32 numpy array.
        Shape: (samples,) for mono, (2, samples) for stereo.
    sr : int
        Sample rate of the audio data.
    format : str, optional
        Output format string. Default is ``"wav"``.
        Supported values depend on soundfile: ``"wav"``, ``"flac"``, ``"ogg"``, etc.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the audio array or sample rate is invalid.
    RuntimeError
        If the file could not be written.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("input.wav")
    >>> ap.save("output.flac", y, sr, format="flac")
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required for saving audio. "
            "Install it with: pip install soundfile"
        ) from e

    if not isinstance(y, np.ndarray):
        raise ValueError(
            f"Expected numpy array for 'y', got {type(y).__name__}."
        )
    if sr <= 0:
        raise ValueError(f"Sample rate must be a positive integer, got {sr}.")

    # soundfile expects shape (samples,) or (samples, channels)
    y_out = y.astype(np.float32)
    if y_out.ndim == 2:
        # librosa: (channels, samples) -> soundfile: (samples, channels)
        y_out = y_out.T

    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    try:
        sf.write(path, y_out, sr, format=format.upper())
    except Exception as e:
        raise RuntimeError(
            f"Failed to save audio to '{path}'. "
            f"Check that the format '{format}' is supported.\nOriginal error: {e}"
        ) from e


def record(duration: float = 5.0, sr: int = 44100):
    """
    Record audio from the default system microphone.

    Parameters
    ----------
    duration : float, optional
        Recording duration in seconds. Default is 5.0 seconds.
    sr : int, optional
        Sample rate for recording. Default is 44100 Hz.

    Returns
    -------
    y : np.ndarray
        Recorded audio as a float32 1-D mono array.
    sr : int
        Sample rate used for recording.

    Raises
    ------
    RuntimeError
        If no microphone is found or recording fails.

    Example
    -------
    >>> import audiopy as ap
    >>> print("Recording for 3 seconds...")
    >>> y, sr = ap.record(duration=3, sr=44100)
    >>> ap.save("recording.wav", y, sr)
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise ImportError(
            "sounddevice is required for recording. "
            "Install it with: pip install sounddevice"
        ) from e

    if duration <= 0:
        raise ValueError(f"Recording duration must be positive, got {duration}.")
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}.")

    try:
        print(f"[audiopy] Recording for {duration:.1f} seconds at {sr} Hz...")
        recording = sd.rec(
            int(duration * sr),
            samplerate=sr,
            channels=1,
            dtype="float32",
        )
        sd.wait()  # Block until recording is complete
        print("[audiopy] Recording complete.")
        y = recording.flatten().astype(np.float32)
        return y, int(sr)
    except Exception as e:
        raise RuntimeError(
            f"Failed to record audio. "
            "Make sure a microphone is connected and permissions are granted.\n"
            f"Original error: {e}"
        ) from e


def from_url(url: str, sr: int = 22050):
    """
    Download audio from a URL and load it into a numpy array.

    The file is downloaded to a temporary location, loaded, and then deleted.
    Supports any URL pointing to a valid audio file (wav, mp3, etc.).

    Parameters
    ----------
    url : str
        URL of the audio file to download (e.g., https://example.com/audio.wav).
    sr : int, optional
        Target sample rate. Default is 22050 Hz.

    Returns
    -------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the returned audio.

    Raises
    ------
    ValueError
        If the URL is empty or invalid.
    RuntimeError
        If the download or decoding fails.

    Example
    -------
    >>> import audiopy as ap
    >>> url = "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav"
    >>> y, sr = ap.from_url(url)
    >>> print(f"Downloaded {len(y)/sr:.2f} seconds")
    """
    try:
        import requests
    except ImportError as e:
        raise ImportError(
            "requests is required for downloading audio. "
            "Install it with: pip install requests"
        ) from e

    if not url or not url.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid URL: '{url}'. Must start with 'http://' or 'https://'."
        )

    try:
        print(f"[audiopy] Downloading audio from: {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download audio from '{url}'.\nOriginal error: {e}"
        ) from e

    # Detect extension from URL or Content-Type
    ext = ".wav"
    url_lower = url.lower()
    for candidate in [".mp3", ".flac", ".ogg", ".m4a", ".wav"]:
        if candidate in url_lower:
            ext = candidate
            break
    else:
        content_type = response.headers.get("Content-Type", "")
        if "mpeg" in content_type or "mp3" in content_type:
            ext = ".mp3"
        elif "flac" in content_type:
            ext = ".flac"
        elif "ogg" in content_type:
            ext = ".ogg"

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

        y, sr_out = load(tmp_path, sr=sr)
        return y, sr_out
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode audio downloaded from '{url}'.\nOriginal error: {e}"
        ) from e
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def info(path: str) -> dict:
    """
    Return metadata about an audio file without loading the full audio.

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - ``"duration"``     : float, duration in seconds
        - ``"sample_rate"``  : int, sample rate in Hz
        - ``"channels"``     : int, number of audio channels
        - ``"format"``       : str, file format (e.g., ``"WAV"``)
        - ``"file_size"``    : int, file size in bytes
        - ``"subtype"``      : str, sample format (e.g., ``"PCM_16"``)
        - ``"frames"``       : int, total number of audio frames

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the metadata cannot be read.

    Example
    -------
    >>> import audiopy as ap
    >>> meta = ap.info("audio.wav")
    >>> print(meta)
    {'duration': 3.5, 'sample_rate': 44100, 'channels': 2, 'format': 'WAV', ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Audio file not found: '{path}'. "
            "Please check the path and try again."
        )

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required for reading audio info. "
            "Install it with: pip install soundfile"
        ) from e

    try:
        sf_info = sf.info(path)
        file_size = os.path.getsize(path)
        return {
            "duration": float(sf_info.duration),
            "sample_rate": int(sf_info.samplerate),
            "channels": int(sf_info.channels),
            "format": str(sf_info.format),
            "subtype": str(sf_info.subtype),
            "frames": int(sf_info.frames),
            "file_size": int(file_size),
        }
    except Exception as e:
        raise RuntimeError(
            f"Failed to read metadata from '{path}'.\nOriginal error: {e}"
        ) from e
