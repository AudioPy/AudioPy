"""
audiopy.models.classifier — Audio Classification Module
=========================================================
Sound event classification and speech detection using HuggingFace
Audio Spectrogram Transformer (AST) and related models.
All models are lazy-loaded on first call.
"""

import numpy as np

__all__ = [
    "classify",
    "is_speech",
    "classify_music_genre",
]

# Module-level model cache
_pipeline_cache: dict = {}


def _get_pipeline(model: str, task: str = "audio-classification"):
    """Retrieve or create a cached HuggingFace audio classification pipeline."""
    cache_key = (model, task)
    if cache_key not in _pipeline_cache:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers is required for audio classification. "
                "Install with: pip install transformers torch"
            ) from e

        print(f"[audiopy] Loading classifier model '{model}' "
              "(this may take a moment on first run)...")
        try:
            _pipeline_cache[cache_key] = pipeline(task, model=model)
            print(f"[audiopy] Classifier model '{model}' loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model}'.\n"
                f"Original error: {e}"
            ) from e
    return _pipeline_cache[cache_key]


def _prepare_audio_for_classification(y: np.ndarray, sr: int) -> dict:
    """
    Internal: prepare audio for HuggingFace audio classification.
    Converts to mono float32 at the required sample rate (16 kHz).
    """
    target_sr = 16000
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
                    f"Install librosa or scipy.\nOriginal error: {e}"
                ) from e

    return {"raw": y, "sampling_rate": target_sr}


def classify(
    y: np.ndarray,
    sr: int,
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    top_k: int = 5,
) -> list:
    """
    Classify the sound event in an audio signal.

    Uses the Audio Spectrogram Transformer (AST) fine-tuned on AudioSet,
    a large-scale audio dataset with 527 sound event categories including
    speech, music, animals, environmental sounds, and more.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.
    model : str, optional
        HuggingFace model ID for audio classification.
        Default is ``"MIT/ast-finetuned-audioset-10-10-0.4593"``.
    top_k : int, optional
        Number of top predictions to return. Default is 5.

    Returns
    -------
    list of dict
        A list of top-k predictions, sorted by score (descending). Each
        dictionary contains:

        - ``"label"`` : str, predicted sound category
        - ``"score"`` : float, confidence score in [0, 1]

        Example: ``[{"label": "Dog", "score": 0.97}, ...]``

    Raises
    ------
    ValueError
        If top_k is not a positive integer.
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If classification fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("dog_barking.wav")
    >>> results = ap.classify(y, sr, top_k=3)
    >>> for r in results:
    ...     print(f"{r['label']}: {r['score']:.2%}")
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    pipe = _get_pipeline(model, "audio-classification")
    audio_input = _prepare_audio_for_classification(y, sr)

    try:
        results = pipe(audio_input, top_k=top_k)
        return [{"label": r["label"], "score": float(r["score"])} for r in results]
    except Exception as e:
        raise RuntimeError(
            f"Audio classification failed.\nOriginal error: {e}"
        ) from e


def is_speech(y: np.ndarray, sr: int) -> bool:
    """
    Detect whether an audio clip contains speech.

    Uses the AudioSet classifier to check if speech-related categories
    appear in the top predictions. Returns ``True`` if speech is detected
    with a confidence score above a threshold.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.

    Returns
    -------
    bool
        ``True`` if the audio is classified as containing speech,
        ``False`` otherwise.

    Raises
    ------
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If classification fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("voice_recording.wav")
    >>> if ap.is_speech(y, sr):
    ...     print("Speech detected!")
    ... else:
    ...     print("No speech found.")
    """
    SPEECH_LABELS = {
        "speech", "narration, monologue", "conversation", "babbling",
        "speech synthesizer", "shout", "bellow", "whoop", "whispering",
        "laughter", "baby laughter", "crowd", "male speech, man speaking",
        "female speech, woman speaking", "child speech, kid speaking",
    }

    try:
        results = classify(y, sr, top_k=10)
        for r in results:
            if r["label"].lower() in {s.lower() for s in SPEECH_LABELS}:
                if r["score"] > 0.1:
                    return True
        return False
    except Exception as e:
        raise RuntimeError(
            f"Speech detection failed.\nOriginal error: {e}"
        ) from e


def classify_music_genre(y: np.ndarray, sr: int) -> list:
    """
    Classify the music genre of an audio signal.

    Uses the MAEST (Music Audio Efficient Spectrogram Transformer) model
    fine-tuned on the Discogs music dataset with over 400 genre labels.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array. For best results,
        use at least 10-30 seconds of audio.
    sr : int
        Sample rate of the input audio in Hz.

    Returns
    -------
    list of dict
        Top-5 genre predictions, sorted by score. Each dict contains:

        - ``"label"`` : str, music genre name
        - ``"score"`` : float, confidence score in [0, 1]

        Example: ``[{"label": "Electronic---Techno", "score": 0.82}, ...]``

    Raises
    ------
    ImportError
        If transformers or torch is not installed.
    RuntimeError
        If genre classification fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("song.wav")
    >>> genres = ap.classify_music_genre(y, sr)
    >>> print(f"Top genre: {genres[0]['label']} ({genres[0]['score']:.1%})")
    """
    model = "mtg-upf/discogs-maest-30s-pw-129e"

    try:
        pipe = _get_pipeline(model, "audio-classification")
        audio_input = _prepare_audio_for_classification(y, sr)
        results = pipe(audio_input, top_k=5)
        return [{"label": r["label"], "score": float(r["score"])} for r in results]
    except Exception as e:
        raise RuntimeError(
            f"Music genre classification failed.\nOriginal error: {e}"
        ) from e
