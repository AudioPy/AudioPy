"""
audiopy.models.emotion — Speech Emotion Recognition Module
============================================================
Detect emotional states from speech audio using fine-tuned Wav2Vec2 models
from HuggingFace. All models are lazy-loaded on first call.
"""

import numpy as np

__all__ = [
    "detect_emotion",
    "dominant_emotion",
]

# Module-level model cache
_pipeline_cache: dict = {}

# Default emotion recognition model
_DEFAULT_MODEL = "superb/wav2vec2-base-superb-er"


def _get_pipeline(model: str):
    """Retrieve or create a cached HuggingFace emotion recognition pipeline."""
    if model not in _pipeline_cache:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers is required for emotion recognition. "
                "Install with: pip install transformers torch"
            ) from e

        print(f"[audiopy] Loading emotion model '{model}' "
              "(this may take a moment on first run)...")
        try:
            _pipeline_cache[model] = pipeline(
                "audio-classification",
                model=model,
            )
            print(f"[audiopy] Emotion model '{model}' loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load emotion model '{model}'.\n"
                f"Check your internet connection.\nOriginal error: {e}"
            ) from e
    return _pipeline_cache[model]


def _prepare_audio(y: np.ndarray, sr: int, target_sr: int = 16000) -> dict:
    """Internal: resample to target_sr, convert to mono float32."""
    if y.ndim == 2:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)

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
                    f"Failed to resample audio to {target_sr} Hz.\n"
                    f"Original error: {e}"
                ) from e

    return {"raw": y, "sampling_rate": target_sr}


def detect_emotion(
    y: np.ndarray,
    sr: int,
    model: str = _DEFAULT_MODEL,
) -> list:
    """
    Detect emotional content in a speech audio signal.

    Uses a Wav2Vec2 model fine-tuned for speech emotion recognition (SER).
    The model classifies audio into emotion categories such as happy, sad,
    angry, neutral, fearful, disgusted, and surprised.

    Audio is resampled to 16 kHz internally before processing.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array. Best results with
        2–10 seconds of speech audio.
    sr : int
        Sample rate of the input audio in Hz.
    model : str, optional
        HuggingFace model ID for emotion recognition. Default is
        ``"superb/wav2vec2-base-superb-er"``.

    Returns
    -------
    list of dict
        All predicted emotion labels with confidence scores, sorted by
        score (descending). Each dictionary contains:

        - ``"label"`` : str, emotion category (e.g., ``"happy"``)
        - ``"score"`` : float, confidence score in [0, 1]

        Example:
        ``[{"label": "happy", "score": 0.91}, {"label": "neutral", "score": 0.06}, ...]``

    Raises
    ------
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If emotion detection fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("happy_speech.wav")
    >>> emotions = ap.detect_emotion(y, sr)
    >>> for e in emotions:
    ...     print(f"{e['label']}: {e['score']:.1%}")
    happy: 91.2%
    neutral: 5.8%
    angry: 1.9%
    ...
    """
    pipe = _get_pipeline(model)
    audio_input = _prepare_audio(y, sr, target_sr=16000)

    try:
        results = pipe(audio_input)
        # Normalize and sort by score
        emotion_list = [
            {"label": r["label"].lower(), "score": float(r["score"])}
            for r in results
        ]
        emotion_list.sort(key=lambda x: x["score"], reverse=True)
        return emotion_list
    except Exception as e:
        raise RuntimeError(
            f"Emotion detection failed.\nOriginal error: {e}"
        ) from e


def dominant_emotion(
    y: np.ndarray,
    sr: int,
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    Return the dominant (highest-confidence) emotion detected in speech.

    A convenience wrapper around ``detect_emotion`` that returns only the
    top emotion label as a plain string.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.
    model : str, optional
        HuggingFace model ID. Default is
        ``"superb/wav2vec2-base-superb-er"``.

    Returns
    -------
    str
        The top predicted emotion label (e.g., ``"happy"``, ``"angry"``,
        ``"sad"``, ``"neutral"``).

    Raises
    ------
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If emotion detection fails or returns empty results.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("voice.wav")
    >>> emotion = ap.dominant_emotion(y, sr)
    >>> print(f"Detected emotion: {emotion}")
    Detected emotion: happy
    """
    try:
        emotions = detect_emotion(y, sr, model=model)
        if not emotions:
            raise RuntimeError(
                "Emotion model returned no predictions. "
                "Ensure the audio contains speech."
            )
        return emotions[0]["label"]
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to determine dominant emotion.\nOriginal error: {e}"
        ) from e
