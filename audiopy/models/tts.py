"""
audiopy.models.tts — Text-to-Speech Module
============================================
Convert text to spoken audio using gTTS (Google TTS) for fast synthesis
and HuggingFace SpeechT5 for high-quality, customizable voice synthesis.
"""

import numpy as np
import io
import tempfile
import os

__all__ = [
    "speak",
    "speak_advanced",
]

# Module-level model cache for SpeechT5
_model_cache: dict = {}


def speak(text: str, language: str = "en", slow: bool = False):
    """
    Convert text to speech using Google Text-to-Speech (gTTS).

    A fast, lightweight TTS engine powered by the Google Translate TTS API.
    Returns audio as a numpy float32 array that can be played or saved with
    other audiopy functions.

    Parameters
    ----------
    text : str
        The text to synthesize into speech. Can be a sentence or longer
        paragraph. Non-English text is supported via the ``language`` parameter.
    language : str, optional
        BCP 47 language code. Default is ``"en"`` (English). Examples:
        ``"fr"`` (French), ``"es"`` (Spanish), ``"ar"`` (Arabic),
        ``"de"`` (German), ``"zh"`` (Chinese).
        See https://gtts.readthedocs.io/en/latest/#supported-languages for
        a full list.
    slow : bool, optional
        If ``True``, synthesizes speech at a slower rate. Useful for language
        learning applications. Default is ``False``.

    Returns
    -------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the returned audio (22050 Hz).

    Raises
    ------
    ValueError
        If text is empty or language is invalid.
    ImportError
        If gTTS is not installed.
    RuntimeError
        If TTS synthesis fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.speak("Hello! Welcome to the audiopy library.", language="en")
    >>> ap.save("hello.wav", y, sr)

    >>> # French TTS:
    >>> y_fr, sr = ap.speak("Bonjour le monde!", language="fr")
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")

    try:
        from gtts import gTTS
    except ImportError as e:
        raise ImportError(
            "gTTS is required for basic TTS. Install with: pip install gTTS"
        ) from e

    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required to decode the TTS audio. "
            "Install with: pip install librosa"
        ) from e

    try:
        tts = gTTS(text=text.strip(), lang=language, slow=slow)
    except ValueError as e:
        raise ValueError(
            f"Invalid language code '{language}' for gTTS.\n"
            f"See supported languages at: https://gtts.readthedocs.io/\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"gTTS initialization failed.\nOriginal error: {e}"
        ) from e

    try:
        # Write to temporary MP3 file, then load with librosa
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        tts.save(tmp_path)
        y, sr_out = librosa.load(tmp_path, sr=22050)
        y = y.astype(np.float32)
        return y, int(sr_out)
    except Exception as e:
        raise RuntimeError(
            f"TTS audio synthesis or loading failed.\nOriginal error: {e}"
        ) from e
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def speak_advanced(
    text: str,
    model: str = "microsoft/speecht5_tts",
):
    """
    Convert text to high-quality speech using Microsoft SpeechT5 via HuggingFace.

    SpeechT5 is a unified-modal encoder-decoder framework pre-trained on
    speech and text data. It produces significantly higher quality audio
    than gTTS with a natural-sounding voice.

    Models are lazy-loaded on the first call. Requires ~1.5 GB of disk space
    for the model weights on first download.

    Parameters
    ----------
    text : str
        The text to synthesize. Best with sentences up to ~100 words.
    model : str, optional
        HuggingFace SpeechT5 model ID. Default is ``"microsoft/speecht5_tts"``.

    Returns
    -------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the synthesized audio (16000 Hz).

    Raises
    ------
    ValueError
        If text is empty.
    ImportError
        If transformers, torch, or datasets is not installed.
    RuntimeError
        If TTS synthesis fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.speak_advanced("This is a high quality text to speech example.")
    >>> ap.save("advanced_tts.wav", y, sr)
    >>> fig = ap.plot_waveform(y, sr, title="SpeechT5 Output")
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")

    try:
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        import torch
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required for advanced TTS. "
            "Install with: pip install transformers torch"
        ) from e

    cache_key = f"speecht5_{model}"
    if cache_key not in _model_cache:
        print(f"[audiopy] Loading SpeechT5 model '{model}' "
              "(first run downloads ~1.5 GB)...")
        try:
            processor = SpeechT5Processor.from_pretrained(model)
            tts_model = SpeechT5ForTextToSpeech.from_pretrained(model)
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            _model_cache[cache_key] = {
                "processor": processor,
                "model": tts_model,
                "vocoder": vocoder,
            }
            print(f"[audiopy] SpeechT5 model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SpeechT5 model '{model}'.\n"
                f"Check your internet connection.\nOriginal error: {e}"
            ) from e

    components = _model_cache[cache_key]
    processor = components["processor"]
    tts_model = components["model"]
    vocoder = components["vocoder"]

    try:
        # Load a default speaker embedding from datasets
        try:
            from datasets import load_dataset
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors", split="validation"
            )
            speaker_embeddings = torch.tensor(
                embeddings_dataset[7306]["xvector"]
            ).unsqueeze(0)
        except Exception:
            # Fallback: use random speaker embeddings
            speaker_embeddings = torch.zeros((1, 512))

        inputs = processor(text=text.strip(), return_tensors="pt")

        with torch.no_grad():
            speech = tts_model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=vocoder,
            )

        y = speech.numpy().astype(np.float32)
        sr_out = 16000
        return y, int(sr_out)
    except Exception as e:
        raise RuntimeError(
            f"SpeechT5 synthesis failed.\nOriginal error: {e}"
        ) from e
