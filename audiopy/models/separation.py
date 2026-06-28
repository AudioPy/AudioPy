"""
audiopy.models.separation — Audio Source Separation Module
============================================================
Separate vocals and instruments from mixed audio using Demucs (Facebook Research).
Falls back to Spleeter if Demucs is unavailable. Models are lazy-loaded.
"""

import numpy as np
import tempfile
import os

__all__ = [
    "separate_vocals",
    "separate_instruments",
]

# Module-level model cache
_model_cache: dict = {}


def _save_temp_wav(y: np.ndarray, sr: int) -> str:
    """Save audio to a temporary WAV file, return the path."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "soundfile is required. Install with: pip install soundfile"
        ) from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    y_out = y.astype(np.float32)
    if y_out.ndim == 2:
        y_out = y_out.T  # (channels, samples) -> (samples, channels)

    sf.write(tmp_path, y_out, sr)
    return tmp_path


def _to_stereo_float32(y: np.ndarray) -> np.ndarray:
    """
    Return audio as a (2, samples) float32 stereo array.
    Handles mono, stereo in either layout, and multi-channel.
    """
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        return np.stack([y, y])  # mono -> (2, N)

    if y.ndim == 2:
        r, c = y.shape
        if r == 2 and c != 2:
            return y  # already (2, N)
        if c == 2 and r != 2:
            return y.T  # (N, 2) -> (2, N)
        if r == 2 and c == 2:
            return y  # ambiguous, assume (2, N)
        # multi-channel -> collapse to mono then duplicate
        mono = y.mean(axis=0)
        return np.stack([mono, mono])

    raise ValueError(f"Unexpected audio shape {y.shape}.")


def _separate_with_demucs(y: np.ndarray, sr: int, model_name: str = "htdemucs") -> dict:
    """
    Separate audio sources using Demucs (v4.x API).
    Returns dict mapping source name -> mono float32 numpy array.
    """
    # Check torch
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for Demucs source separation. "
            "Install with: pip install torch"
        ) from exc

    # Check demucs
    try:
        from demucs.pretrained import get_model  # type: ignore
        from demucs.apply import apply_model     # type: ignore
    except ModuleNotFoundError:
        raise ImportError(
            "demucs is not installed. Install with: pip install demucs"
        )
    except ImportError as exc:
        raise ImportError(
            f"Failed to import demucs (a dependency may be missing).\n"
            f"Original error: {exc}"
        ) from exc

    # Load / cache model
    if model_name not in _model_cache:
        print(f"[audiopy] Loading Demucs model '{model_name}' "
              "(first run downloads ~80-300 MB)...")
        try:
            model = get_model(model_name)
            model.eval()
            _model_cache[model_name] = model
            print(f"[audiopy] Demucs model '{model_name}' loaded.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Demucs model '{model_name}'.\n"
                f"Original error: {exc}"
            ) from exc

    model = _model_cache[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare stereo audio at model sample rate
    y_stereo = _to_stereo_float32(y)
    model_sr = model.samplerate

    if sr != model_sr:
        try:
            import librosa
            y_stereo = np.stack([
                librosa.resample(y_stereo[ch], orig_sr=sr, target_sr=model_sr)
                for ch in range(2)
            ])
        except ImportError:
            try:
                import torchaudio
                t = torch.from_numpy(np.ascontiguousarray(y_stereo))
                y_stereo = torchaudio.functional.resample(
                    t, orig_freq=sr, new_freq=model_sr
                ).numpy()
            except ImportError:
                import warnings
                warnings.warn(
                    f"Cannot resample from {sr} to {model_sr} Hz. "
                    "Install librosa or torchaudio for resampling.",
                    RuntimeWarning, stacklevel=4,
                )

    # Build batch tensor (1, 2, N)
    wav_tensor = torch.from_numpy(
        np.ascontiguousarray(y_stereo)
    ).unsqueeze(0).to(device)

    # Run separation
    try:
        with torch.no_grad():
            out = apply_model(model, wav_tensor, device=device, split=True, overlap=0.25)
    except TypeError:
        # Older Demucs versions may not accept all kwargs
        with torch.no_grad():
            out = apply_model(model, wav_tensor)

    # out: (1, num_sources, 2, N)
    sources_np = out.squeeze(0).cpu().numpy()
    source_names = model.sources

    result = {}
    for i, name in enumerate(source_names):
        result[name] = sources_np[i].mean(axis=0).astype(np.float32)

    # Resample back to original sr
    if sr != model_sr:
        try:
            import librosa
            result = {
                name: librosa.resample(arr, orig_sr=model_sr, target_sr=sr)
                for name, arr in result.items()
            }
        except ImportError:
            pass

    return result


def _separate_with_spleeter(y: np.ndarray, sr: int, stems: int = 2) -> dict:
    """
    Separate audio sources using Spleeter (fallback).
    Note: spleeter.audio.adapter.AudioAdapter was removed in spleeter>=2.3.
    """
    try:
        from spleeter.separator import Separator  # type: ignore
    except ModuleNotFoundError:
        raise ImportError(
            "spleeter is not installed. Install with: pip install spleeter"
        )
    except ImportError as exc:
        raise ImportError(
            f"spleeter is installed but failed to import "
            f"(tensorflow may be missing).\nOriginal error: {exc}"
        ) from exc

    config = f"spleeter:{stems}stems"

    if config not in _model_cache:
        print(f"[audiopy] Loading Spleeter '{config}' model...")
        try:
            _model_cache[config] = Separator(config)
            print("[audiopy] Spleeter model loaded.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise Spleeter '{config}'.\n"
                f"Original error: {exc}"
            ) from exc

    separator = _model_cache[config]

    # Spleeter expects (samples, channels) float32
    y_stereo = _to_stereo_float32(y).T.astype(np.float32)

    try:
        prediction = separator.separate(y_stereo)
        return {
            name: np.mean(audio, axis=-1).astype(np.float32)
            for name, audio in prediction.items()
        }
    except Exception as exc:
        raise RuntimeError(
            f"Spleeter separation failed.\nOriginal error: {exc}"
        ) from exc


def separate_vocals(y: np.ndarray, sr: int) -> dict:
    """
    Separate vocals from the musical accompaniment of a mixed audio signal.

    Attempts to use Facebook's Demucs (``htdemucs``) model first.
    Falls back to Spleeter (2-stem) if Demucs is unavailable.
    The model is downloaded automatically on the first call (~80-300 MB).

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.

    Returns
    -------
    dict
        - ``"vocals"``         : np.ndarray, isolated vocal track
        - ``"accompaniment"``  : np.ndarray, music without vocals

    Raises
    ------
    ImportError
        If neither demucs nor spleeter is installed.
    RuntimeError
        If separation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("song.mp3")
    >>> stems = ap.separate_vocals(y, sr)
    >>> ap.save("vocals.wav", stems["vocals"], sr)
    >>> ap.save("instrumental.wav", stems["accompaniment"], sr)
    """
    sources = None
    last_err = None

    # Try Demucs first
    try:
        sources = _separate_with_demucs(y, sr, model_name="htdemucs")
    except ImportError as exc:
        last_err = exc
        print(f"[audiopy] Demucs unavailable ({exc}), trying Spleeter...")
    except Exception as exc:
        raise RuntimeError(
            f"Vocal separation with Demucs failed.\nOriginal error: {exc}"
        ) from exc

    # Try Spleeter fallback
    if sources is None:
        try:
            spleeter_result = _separate_with_spleeter(y, sr, stems=2)
            return {k: v.astype(np.float32) for k, v in spleeter_result.items()}
        except ImportError as exc:
            raise ImportError(
                "Source separation requires 'demucs' or 'spleeter'.\n"
                "  pip install demucs     (recommended)\n"
                "  pip install spleeter   (fallback)\n"
                f"Last error: {last_err}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Vocal separation failed.\nOriginal error: {exc}"
            ) from exc

    # Build vocals + accompaniment from Demucs output
    vocals = sources.get("vocals", np.zeros(
        y.shape[-1] if y.ndim == 2 else y.shape[0], dtype=np.float32
    ))

    accompaniment_parts = [v for k, v in sources.items() if k != "vocals"]
    if accompaniment_parts:
        min_len = min(len(p) for p in accompaniment_parts)
        accompaniment = sum(p[:min_len] for p in accompaniment_parts).astype(np.float32)
        peak = np.max(np.abs(accompaniment))
        if peak > 1.0:
            accompaniment /= peak
    else:
        accompaniment = np.zeros_like(vocals)

    return {
        "vocals": vocals.astype(np.float32),
        "accompaniment": accompaniment.astype(np.float32),
    }


def separate_instruments(y: np.ndarray, sr: int) -> dict:
    """
    Separate a music track into its constituent instrument stems.

    Uses Demucs ``htdemucs`` 4-stem model. Falls back to Spleeter 4-stem.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.

    Returns
    -------
    dict
        - ``"drums"``   : np.ndarray
        - ``"bass"``    : np.ndarray
        - ``"other"``   : np.ndarray
        - ``"vocals"``  : np.ndarray

    Raises
    ------
    ImportError
        If neither demucs nor spleeter is installed.
    RuntimeError
        If separation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("song.mp3")
    >>> stems = ap.separate_instruments(y, sr)
    >>> ap.save("drums.wav", stems["drums"], sr)
    """
    sources = None
    last_err = None

    try:
        sources = _separate_with_demucs(y, sr, model_name="htdemucs")
    except ImportError as exc:
        last_err = exc
        print(f"[audiopy] Demucs unavailable ({exc}), trying Spleeter 4-stem...")
    except Exception as exc:
        raise RuntimeError(
            f"Instrument separation with Demucs failed.\nOriginal error: {exc}"
        ) from exc

    if sources is None:
        try:
            spleeter_result = _separate_with_spleeter(y, sr, stems=4)
            return {k: v.astype(np.float32) for k, v in spleeter_result.items()}
        except ImportError as exc:
            raise ImportError(
                "Instrument separation requires 'demucs' or 'spleeter'.\n"
                "  pip install demucs     (recommended)\n"
                "  pip install spleeter   (fallback)\n"
                f"Last error: {last_err}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Instrument separation failed.\nOriginal error: {exc}"
            ) from exc

    expected_stems = ["drums", "bass", "other", "vocals"]
    ref_len = len(next(iter(sources.values())))
    result = {}
    for stem in expected_stems:
        if stem in sources:
            result[stem] = sources[stem].astype(np.float32)
        else:
            result[stem] = np.zeros(ref_len, dtype=np.float32)

    return result
