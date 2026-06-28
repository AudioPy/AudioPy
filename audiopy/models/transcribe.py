"""
audiopy.models.transcribe — Speech-to-Text Module
===================================================
Automatic speech recognition (ASR) using HuggingFace Whisper models.
Models are lazy-loaded on first call — no download at import time.
"""

import numpy as np
from typing import Optional

__all__ = [
    "transcribe",
    "transcribe_with_timestamps",
    "translate_to_english",
]

# Module-level model cache (lazy loading)
_pipeline_cache: dict = {}


def _get_pipeline(model: str, task: str = "automatic-speech-recognition"):
    """
    Retrieve or create a cached HuggingFace pipeline for the given model/task.

    Models are downloaded on first call and cached in memory for subsequent calls.
    """
    cache_key = (model, task)
    if cache_key not in _pipeline_cache:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers is required for speech recognition. "
                "Install with: pip install transformers torch"
            ) from e

        print(f"[audiopy] Loading model '{model}' for task '{task}' "
              "(this may take a moment on first run)...")
        try:
            _pipeline_cache[cache_key] = pipeline(task, model=model)
            print(f"[audiopy] Model '{model}' loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model}'.\n"
                f"Check your internet connection and model name.\n"
                f"Original error: {e}"
            ) from e
    return _pipeline_cache[cache_key]


def _prepare_audio(y: np.ndarray, sr: int, target_sr: int = 16000) -> dict:
    """
    Internal: prepare audio for Whisper (resample to 16 kHz, ensure float32 mono).
    Returns a dict in the format expected by HuggingFace pipelines.
    """
    # Convert to mono
    if y.ndim == 2:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)

    # Resample to target sample rate (Whisper requires 16 kHz)
    if sr != target_sr:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(target_sr, sr)
                y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resample audio to {target_sr} Hz. "
                    "Install librosa or scipy.\nOriginal error: {e}"
                ) from e

    return {"raw": y, "sampling_rate": target_sr}


def transcribe(
    y: np.ndarray,
    sr: int,
    model: str = "openai/whisper-base",
    language: Optional[str] = None,
) -> str:
    """
    Transcribe speech from an audio signal to text (Speech-to-Text).

    Uses OpenAI's Whisper model via the HuggingFace transformers library.
    The model is automatically downloaded on the first call. Audio is
    resampled to 16 kHz internally before processing.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.
    model : str, optional
        HuggingFace model ID. Default is ``"openai/whisper-base"``.
        Available sizes: ``"openai/whisper-tiny"``, ``"openai/whisper-small"``,
        ``"openai/whisper-medium"``, ``"openai/whisper-large-v3"``.
    language : str, optional
        Force a specific language for recognition (e.g., ``"french"``,
        ``"arabic"``, ``"spanish"``). If ``None``, language is auto-detected.

    Returns
    -------
    str
        Transcribed text as a string.

    Raises
    ------
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If transcription fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("speech.wav")
    >>> text = ap.transcribe(y, sr)
    >>> print(text)
    "Hello, welcome to the audiopy library."

    >>> # Force French language detection:
    >>> text_fr = ap.transcribe(y, sr, language="french")
    """
    pipe = _get_pipeline(model, "automatic-speech-recognition")
    audio_input = _prepare_audio(y, sr, target_sr=16000)

    generate_kwargs = {}
    if language is not None:
        generate_kwargs["language"] = language.lower()

    try:
        result = pipe(audio_input, generate_kwargs=generate_kwargs)
        return result["text"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Transcription failed.\nOriginal error: {e}"
        ) from e


def transcribe_with_timestamps(
    y: np.ndarray,
    sr: int,
    model: str = "openai/whisper-base",
) -> list:
    """
    Transcribe audio and return word/segment-level timestamps.

    Returns a list of segments with start time, end time, and text for each.
    Requires Whisper's ``return_timestamps=True`` pipeline mode.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.
    model : str, optional
        HuggingFace Whisper model ID. Default is ``"openai/whisper-base"``.

    Returns
    -------
    list of dict
        A list of segment dictionaries, each containing:

        - ``"text"``  : str, transcribed text for this segment
        - ``"start"`` : float, start time in seconds
        - ``"end"``   : float, end time in seconds

        Example: ``[{"text": "Hello world", "start": 0.0, "end": 1.2}, ...]``

    Raises
    ------
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If transcription fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("speech.wav")
    >>> segments = ap.transcribe_with_timestamps(y, sr)
    >>> for seg in segments:
    ...     print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    """
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transformers is required. Install with: pip install transformers torch"
        ) from e

    cache_key = (model, "asr_timestamps")
    if cache_key not in _pipeline_cache:
        print(f"[audiopy] Loading model '{model}' for timestamped transcription...")
        try:
            _pipeline_cache[cache_key] = pipeline(
                "automatic-speech-recognition",
                model=model,
                return_timestamps=True,
            )
            print(f"[audiopy] Model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model}'.\nOriginal error: {e}"
            ) from e

    pipe = _pipeline_cache[cache_key]
    audio_input = _prepare_audio(y, sr, target_sr=16000)

    try:
        result = pipe(audio_input, return_timestamps=True)
        segments = []
        chunks = result.get("chunks", [])
        for chunk in chunks:
            ts = chunk.get("timestamp", (0.0, 0.0))
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start
            segments.append({
                "text": chunk.get("text", "").strip(),
                "start": start,
                "end": end,
            })
        return segments
    except Exception as e:
        raise RuntimeError(
            f"Timestamped transcription failed.\nOriginal error: {e}"
        ) from e


def translate_to_english(
    y: np.ndarray,
    sr: int,
    model: str = "openai/whisper-base",
) -> str:
    """
    Transcribe non-English audio and translate it to English.

    Uses Whisper's built-in multilingual translation capability. The model
    automatically detects the source language and translates to English.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.
    model : str, optional
        HuggingFace Whisper model ID. Default is ``"openai/whisper-base"``.
        Larger models (e.g., ``"openai/whisper-large-v3"``) give better
        translation quality.

    Returns
    -------
    str
        English translation of the transcribed audio as a string.

    Raises
    ------
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If translation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("french_speech.wav")
    >>> text_en = ap.translate_to_english(y, sr)
    >>> print(text_en)
    "Hello, how are you?"
    """
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transformers is required. Install with: pip install transformers torch"
        ) from e

    cache_key = (model, "translation")
    if cache_key not in _pipeline_cache:
        print(f"[audiopy] Loading model '{model}' for translation to English...")
        try:
            _pipeline_cache[cache_key] = pipeline(
                "automatic-speech-recognition",
                model=model,
            )
            print(f"[audiopy] Model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model}'.\nOriginal error: {e}"
            ) from e

    pipe = _pipeline_cache[cache_key]
    audio_input = _prepare_audio(y, sr, target_sr=16000)

    try:
        result = pipe(audio_input, generate_kwargs={"task": "translate"})
        return result["text"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Translation to English failed.\nOriginal error: {e}"
        ) from e
