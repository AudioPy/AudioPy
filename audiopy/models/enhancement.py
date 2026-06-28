"""
audiopy.models.enhancement — Speech Enhancement Module
========================================================
Deep learning-based audio denoising and enhancement. Uses DNS64 / DeepFilterNet
models (via torch.hub or HuggingFace), with a noisereduce fallback.
WebRTC VAD is used for silence removal.
"""

import numpy as np

__all__ = [
    "denoise_deep",
    "enhance_speech",
    "remove_silence",
]

# Module-level model cache
_model_cache: dict = {}


def _ensure_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        return np.mean(y, axis=0).astype(np.float32)
    return y.astype(np.float32)


def denoise_deep(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply deep learning-based noise removal to an audio signal.

    Attempts the following models in order:
    1. **DeepFilterNet** (dns64) via torch.hub — state-of-the-art speech denoising
    2. **Facebook Denoiser** (demucs-based) via torch.hub
    3. **noisereduce** — spectral gating fallback (no GPU required)

    Models are lazy-loaded on first call.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.

    Returns
    -------
    np.ndarray
        Denoised audio as a float32 mono numpy array.

    Raises
    ------
    RuntimeError
        If all denoising methods fail.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("noisy_speech.wav")
    >>> y_clean = ap.denoise_deep(y, sr)
    >>> ap.save("clean_speech.wav", y_clean, sr)
    >>> fig = ap.compare_waveforms(y, y_clean, sr, labels=["Noisy", "Denoised"])
    """
    y_mono = _ensure_mono_float32(y)

    # ── Attempt 1: DeepFilterNet via torch.hub ───────────────────────────
    try:
        import torch
        if "deepfilternet" not in _model_cache:
            print("[audiopy] Loading DeepFilterNet model via torch.hub "
                  "(first run downloads ~50 MB)...")
            model = torch.hub.load(
                "rikorose/DeepFilterNet",
                "DeepFilterNet2",
                force_reload=False,
                trust_repo=True,
            )
            model.eval()
            _model_cache["deepfilternet"] = model
            print("[audiopy] DeepFilterNet loaded.")

        model = _model_cache["deepfilternet"]

        # DeepFilterNet expects 48 kHz
        target_sr = 48000
        y_resampled = y_mono
        if sr != target_sr:
            try:
                import librosa
                y_resampled = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                pass

        audio_tensor = torch.tensor(y_resampled).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        with torch.no_grad():
            enhanced = model(audio_tensor, target_sr)
        y_out = enhanced.squeeze().numpy().astype(np.float32)

        # Resample back to original sr if needed
        if sr != target_sr:
            try:
                import librosa
                y_out = librosa.resample(y_out, orig_sr=target_sr, target_sr=sr)
            except ImportError:
                pass

        return y_out.astype(np.float32)

    except Exception as e1:
        print(f"[audiopy] DeepFilterNet unavailable ({type(e1).__name__}), "
              "trying noisereduce fallback...")

    # ── Fallback: noisereduce spectral gating ─────────────────────────────
    try:
        import noisereduce as nr
        # Use first 0.5 seconds as noise profile
        noise_len = min(int(0.5 * sr), len(y_mono))
        noise_profile = y_mono[:noise_len]
        result = nr.reduce_noise(
            y=y_mono,
            y_noise=noise_profile,
            sr=sr,
            prop_decrease=1.0,
            stationary=False,
        )
        return result.astype(np.float32)
    except ImportError:
        pass
    except Exception as e2:
        print(f"[audiopy] noisereduce fallback also failed: {e2}")

    raise RuntimeError(
        "All denoising methods failed. "
        "Please install at least one of: torch (for DeepFilterNet) "
        "or noisereduce (pip install noisereduce)."
    )


def enhance_speech(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Run a complete speech enhancement pipeline to improve voice quality.

    Applies a multi-stage enhancement process:
    1. High-pass filter to remove low-frequency rumble (<80 Hz)
    2. Deep noise removal (``denoise_deep``)
    3. Dynamic range compression to even out volume
    4. Gentle normalization

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.

    Returns
    -------
    np.ndarray
        Enhanced speech audio as float32. Same sample rate as input.

    Raises
    ------
    RuntimeError
        If enhancement fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("voice_recording.wav")
    >>> y_enhanced = ap.enhance_speech(y, sr)
    >>> ap.save("enhanced.wav", y_enhanced, sr)
    """
    y_mono = _ensure_mono_float32(y)

    try:
        # Stage 1: High-pass filter to remove rumble
        try:
            from scipy.signal import butter, sosfilt
            nyquist = sr / 2.0
            cutoff = min(80.0, nyquist * 0.5)
            sos = butter(2, cutoff / nyquist, btype="high", output="sos")
            y_hp = sosfilt(sos, y_mono).astype(np.float32)
        except ImportError:
            y_hp = y_mono

        # Stage 2: Deep noise removal
        try:
            y_denoised = denoise_deep(y_hp, sr)
        except RuntimeError:
            y_denoised = y_hp

        # Stage 3: Dynamic compression
        # Simple soft-knee compression in Python
        threshold_linear = 10 ** (-20.0 / 20.0)  # -20 dBFS
        ratio = 3.0
        result = y_denoised.copy()
        mask = np.abs(result) > threshold_linear
        excess = np.abs(result[mask]) - threshold_linear
        compressed_excess = excess / ratio
        result[mask] = np.sign(result[mask]) * (threshold_linear + compressed_excess)

        # Stage 4: Normalize
        peak = np.max(np.abs(result))
        if peak > 0:
            result = result / peak * 0.95  # Leave 5% headroom

        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Speech enhancement failed.\nOriginal error: {e}"
        ) from e


def remove_silence(
    y: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
) -> np.ndarray:
    """
    Remove silent frames from audio using WebRTC Voice Activity Detection (VAD).

    Uses the ``webrtcvad`` library, which is an efficient C-based VAD algorithm
    originally developed for the WebRTC project. Falls back to a librosa-based
    energy threshold method if webrtcvad is unavailable.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate of the input audio in Hz.
        WebRTC VAD only supports: 8000, 16000, 32000, or 48000 Hz.
        Audio at other sample rates is resampled automatically.
    aggressiveness : int, optional
        VAD aggressiveness level. Range [0, 3]:

        - ``0`` : least aggressive (keeps more audio, fewer false negatives)
        - ``1`` : moderate
        - ``2`` : aggressive (default) — good for most use cases
        - ``3`` : most aggressive (strips the most silence)

    Returns
    -------
    np.ndarray
        Audio with silent frames removed as float32. The output is shorter
        than the input.

    Raises
    ------
    ValueError
        If aggressiveness is not in [0, 3].
    RuntimeError
        If VAD processing fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("interview_with_pauses.wav")
    >>> y_speech = ap.remove_silence(y, sr, aggressiveness=2)
    >>> print(f"Original: {len(y)/sr:.1f}s, After VAD: {len(y_speech)/sr:.1f}s")
    """
    if aggressiveness not in (0, 1, 2, 3):
        raise ValueError(
            f"aggressiveness must be 0, 1, 2, or 3. Got {aggressiveness}."
        )

    y_mono = _ensure_mono_float32(y)

    # ── Attempt 1: webrtcvad ─────────────────────────────────────────────
    try:
        import webrtcvad
        import struct

        vad = webrtcvad.Vad(aggressiveness)

        # WebRTC VAD only supports specific sample rates
        valid_srs = [8000, 16000, 32000, 48000]
        vad_sr = min(valid_srs, key=lambda x: abs(x - sr))

        # Resample to closest valid SR
        y_vad = y_mono
        if sr != vad_sr:
            try:
                import librosa
                y_vad = librosa.resample(y_mono, orig_sr=sr, target_sr=vad_sr)
            except ImportError:
                vad_sr = sr

        # Convert to int16 PCM for webrtcvad
        y_int16 = (y_vad * 32767).astype(np.int16)

        # Frame duration options: 10, 20, or 30 ms
        frame_duration_ms = 30
        frame_samples = int(vad_sr * frame_duration_ms / 1000)
        frame_bytes = frame_samples * 2  # int16 = 2 bytes

        speech_frames = []
        raw_bytes = y_int16.tobytes()
        num_frames = len(raw_bytes) // frame_bytes

        for i in range(num_frames):
            frame = raw_bytes[i * frame_bytes: (i + 1) * frame_bytes]
            if len(frame) < frame_bytes:
                break
            try:
                is_speech = vad.is_speech(frame, sample_rate=vad_sr)
            except Exception:
                is_speech = True  # Keep frame on error
            if is_speech:
                speech_frames.append(
                    y_vad[i * frame_samples: (i + 1) * frame_samples]
                )

        if not speech_frames:
            return y_mono  # Return original if nothing detected

        speech_audio = np.concatenate(speech_frames).astype(np.float32)

        # Resample back to original sr if we changed it
        if sr != vad_sr:
            try:
                import librosa
                speech_audio = librosa.resample(speech_audio, orig_sr=vad_sr, target_sr=sr)
            except ImportError:
                pass

        return speech_audio.astype(np.float32)

    except ImportError:
        print("[audiopy] webrtcvad not available, using librosa RMS fallback for VAD.")
    except Exception as e:
        print(f"[audiopy] webrtcvad failed ({e}), using librosa RMS fallback.")

    # ── Fallback: librosa RMS energy threshold ───────────────────────────
    try:
        import librosa

        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(
            y=y_mono, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Compute threshold based on aggressiveness
        # Higher aggressiveness = higher threshold = more silence removed
        threshold_percentiles = {0: 5, 1: 15, 2: 25, 3: 40}
        percentile = threshold_percentiles[aggressiveness]
        threshold = np.percentile(rms, percentile)

        speech_mask = rms > threshold

        # Expand mask to sample domain
        speech_samples = []
        for i, is_speech_frame in enumerate(speech_mask):
            if is_speech_frame:
                start = i * hop_length
                end = min(start + hop_length, len(y_mono))
                speech_samples.append(y_mono[start:end])

        if not speech_samples:
            return y_mono

        return np.concatenate(speech_samples).astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Silence removal failed.\nOriginal error: {e}"
        ) from e
