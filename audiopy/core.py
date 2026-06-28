"""
audiopy.core — Core Audio Processing Module
=============================================
Fundamental audio manipulation operations: resampling, channel conversion,
trimming, padding, silence splitting, concatenation, mixing, and normalization.
"""

import numpy as np

__all__ = [
    "resample",
    "mono",
    "stereo",
    "trim",
    "pad",
    "split_on_silence",
    "concatenate",
    "mix",
    "normalize",
    "duration",
]


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample an audio array from one sample rate to another.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series. Shape: (samples,) or (channels, samples).
    orig_sr : int
        Original sample rate in Hz.
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    np.ndarray
        Resampled audio as a float32 numpy array.

    Raises
    ------
    ValueError
        If sample rates are not positive integers.
    RuntimeError
        If resampling fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav", sr=44100)
    >>> y_16k = ap.resample(y, orig_sr=44100, target_sr=16000)
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required for resampling. "
            "Install with: pip install librosa"
        ) from e

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"Sample rates must be positive integers. "
            f"Got orig_sr={orig_sr}, target_sr={target_sr}."
        )
    if orig_sr == target_sr:
        return y.astype(np.float32)

    try:
        if y.ndim == 2:
            # Resample each channel separately
            channels = [
                librosa.resample(y[i], orig_sr=orig_sr, target_sr=target_sr)
                for i in range(y.shape[0])
            ]
            return np.stack(channels).astype(np.float32)
        return librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    except Exception as e:
        raise RuntimeError(
            f"Failed to resample audio from {orig_sr} Hz to {target_sr} Hz.\n"
            f"Original error: {e}"
        ) from e


def mono(y: np.ndarray) -> np.ndarray:
    """
    Convert a stereo (or multi-channel) audio array to mono.

    Mono audio is returned unchanged. For stereo, channels are averaged.

    Parameters
    ----------
    y : np.ndarray
        Audio array. Shape: (samples,) for mono or (channels, samples) for multi-channel.

    Returns
    -------
    np.ndarray
        1-D float32 mono audio array of shape (samples,).

    Raises
    ------
    ValueError
        If the input array has an unexpected shape.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("stereo.wav")
    >>> y_mono = ap.mono(y)
    >>> print(y_mono.ndim)  # 1
    """
    if not isinstance(y, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(y).__name__}.")

    if y.ndim == 1:
        return y.astype(np.float32)
    elif y.ndim == 2:
        # (channels, samples) -> average channels
        return np.mean(y, axis=0).astype(np.float32)
    else:
        raise ValueError(
            f"Unexpected audio shape {y.shape}. "
            "Expected 1-D (samples,) or 2-D (channels, samples)."
        )


def stereo(y: np.ndarray) -> np.ndarray:
    """
    Convert a mono audio array to stereo by duplicating the channel.

    Stereo input is returned unchanged.

    Parameters
    ----------
    y : np.ndarray
        Audio array. Shape: (samples,) for mono or (2, samples) for stereo.

    Returns
    -------
    np.ndarray
        2-D float32 stereo array of shape (2, samples).

    Raises
    ------
    ValueError
        If the input array shape is unsupported.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("mono.wav")
    >>> y_stereo = ap.stereo(y)
    >>> print(y_stereo.shape)  # (2, N)
    """
    if not isinstance(y, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(y).__name__}.")

    if y.ndim == 1:
        return np.stack([y, y]).astype(np.float32)
    elif y.ndim == 2 and y.shape[0] == 2:
        return y.astype(np.float32)
    elif y.ndim == 2:
        raise ValueError(
            f"Expected mono (samples,) or stereo (2, samples), "
            f"but got shape {y.shape}."
        )
    else:
        raise ValueError(
            f"Unexpected audio shape {y.shape}. "
            "Expected 1-D (samples,) or 2-D (channels, samples)."
        )


def trim(y: np.ndarray, sr: int, top_db: float = 20.0) -> np.ndarray:
    """
    Trim leading and trailing silence from an audio signal.

    Uses librosa's ``trim`` function which removes portions below a
    threshold relative to the signal's peak amplitude.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz (used for internal frame calculations).
    top_db : float, optional
        Threshold (in dB below peak) below which audio is considered silent.
        Default is 20.0 dB. Higher values trim more aggressively.

    Returns
    -------
    np.ndarray
        Trimmed audio as a float32 numpy array.

    Raises
    ------
    RuntimeError
        If trimming fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("padded_speech.wav")
    >>> y_trimmed = ap.trim(y, sr, top_db=25)
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = mono(y) if y.ndim > 1 else y.astype(np.float32)

    try:
        y_trimmed, _ = librosa.effects.trim(y_mono, top_db=top_db)
        return y_trimmed.astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Failed to trim audio.\nOriginal error: {e}"
        ) from e


def pad(y: np.ndarray, target_length: int, mode: str = "constant") -> np.ndarray:
    """
    Pad (or truncate) an audio array to a specific length.

    If the audio is shorter than ``target_length``, it is padded at the end.
    If longer, it is truncated.

    Parameters
    ----------
    y : np.ndarray
        Audio array. Shape: (samples,) or (channels, samples).
    target_length : int
        Desired output length in samples.
    mode : str, optional
        Numpy pad mode. ``"constant"`` (zero-pad, default), ``"wrap"``
        (loop audio), ``"reflect"``, etc. See ``np.pad`` for all options.

    Returns
    -------
    np.ndarray
        Padded/truncated audio as a float32 numpy array, same number of
        dimensions as input.

    Raises
    ------
    ValueError
        If target_length is not a positive integer.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("short.wav")
    >>> y_padded = ap.pad(y, target_length=sr * 10)  # pad to 10 seconds
    """
    if target_length <= 0:
        raise ValueError(
            f"target_length must be a positive integer, got {target_length}."
        )
    if not isinstance(y, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(y).__name__}.")

    try:
        if y.ndim == 1:
            current_length = y.shape[0]
            if current_length >= target_length:
                return y[:target_length].astype(np.float32)
            pad_width = target_length - current_length
            return np.pad(y, (0, pad_width), mode=mode).astype(np.float32)
        elif y.ndim == 2:
            current_length = y.shape[1]
            if current_length >= target_length:
                return y[:, :target_length].astype(np.float32)
            pad_width = target_length - current_length
            return np.pad(y, ((0, 0), (0, pad_width)), mode=mode).astype(np.float32)
        else:
            raise ValueError(f"Unexpected audio shape {y.shape}.")
    except (ValueError, TypeError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to pad audio to length {target_length}.\nOriginal error: {e}"
        ) from e


def split_on_silence(
    y: np.ndarray,
    sr: int,
    min_silence_len: int = 500,
    silence_thresh: float = -40.0,
) -> list:
    """
    Split audio into non-silent segments by detecting silence gaps.

    Parameters
    ----------
    y : np.ndarray
        Mono audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    min_silence_len : int, optional
        Minimum length of a silence region in milliseconds. Default is 500 ms.
    silence_thresh : float, optional
        Audio below this dBFS level is considered silence. Default is -40 dBFS.

    Returns
    -------
    list of np.ndarray
        List of non-silent audio chunks as float32 numpy arrays.

    Raises
    ------
    RuntimeError
        If audio splitting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("speech_with_pauses.wav")
    >>> segments = ap.split_on_silence(y, sr, min_silence_len=300)
    >>> print(f"Found {len(segments)} speech segments")
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    y_mono = mono(y) if y.ndim > 1 else y.astype(np.float32)

    try:
        # Convert silence threshold from dBFS to linear amplitude
        # silence_thresh dBFS: 10^(thresh/20) relative to peak amplitude
        # We compute RMS in hop-sized frames and compare to threshold
        frame_length = int(sr * (min_silence_len / 1000.0))
        hop_length = frame_length // 4

        rms = librosa.feature.rms(y=y_mono, frame_length=max(frame_length, 512), hop_length=max(hop_length, 128))[0]
        # Convert rms to dBFS
        rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) if np.max(rms) > 0 else 1.0)

        is_silent = rms_db < silence_thresh

        # Map frames back to samples
        frames_to_samples = lambda f: librosa.frames_to_samples(f, hop_length=max(hop_length, 128))

        segments = []
        in_segment = False
        seg_start = 0

        for i, silent in enumerate(is_silent):
            sample_pos = frames_to_samples(i)
            if not silent and not in_segment:
                seg_start = sample_pos
                in_segment = True
            elif silent and in_segment:
                seg_end = sample_pos
                segment = y_mono[seg_start:seg_end]
                if len(segment) > 0:
                    segments.append(segment.astype(np.float32))
                in_segment = False

        # Capture trailing segment
        if in_segment:
            segment = y_mono[seg_start:]
            if len(segment) > 0:
                segments.append(segment.astype(np.float32))

        return segments

    except Exception as e:
        raise RuntimeError(
            f"Failed to split audio on silence.\nOriginal error: {e}"
        ) from e


def concatenate(audio_list: list, sr: int) -> np.ndarray:
    """
    Concatenate a list of audio arrays into a single array.

    All arrays must be mono (1-D) or all stereo (2-D with same channel count).

    Parameters
    ----------
    audio_list : list of np.ndarray
        List of audio arrays to concatenate in order.
    sr : int
        Sample rate in Hz (used for validation context only).

    Returns
    -------
    np.ndarray
        Single concatenated audio array as float32.

    Raises
    ------
    ValueError
        If the list is empty or arrays have incompatible shapes.
    RuntimeError
        If concatenation fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y1, sr = ap.load("part1.wav")
    >>> y2, _ = ap.load("part2.wav", sr=sr)
    >>> y_full = ap.concatenate([y1, y2], sr)
    """
    if not audio_list:
        raise ValueError("audio_list is empty. Provide at least one audio array.")

    if not all(isinstance(a, np.ndarray) for a in audio_list):
        raise ValueError("All elements in audio_list must be numpy arrays.")

    # Validate consistent dimensionality
    ndims = set(a.ndim for a in audio_list)
    if len(ndims) > 1:
        raise ValueError(
            "All audio arrays must have the same number of dimensions "
            "(all mono or all stereo)."
        )

    try:
        arrays = [a.astype(np.float32) for a in audio_list]
        if arrays[0].ndim == 1:
            return np.concatenate(arrays, axis=0)
        else:
            return np.concatenate(arrays, axis=1)
    except Exception as e:
        raise RuntimeError(
            f"Failed to concatenate audio arrays.\nOriginal error: {e}"
        ) from e


def mix(
    y1: np.ndarray,
    y2: np.ndarray,
    weight1: float = 0.5,
    weight2: float = 0.5,
) -> np.ndarray:
    """
    Mix two audio signals together with optional weighting.

    The signals are zero-padded to the length of the longer one before mixing.

    Parameters
    ----------
    y1 : np.ndarray
        First audio signal (float32).
    y2 : np.ndarray
        Second audio signal (float32).
    weight1 : float, optional
        Weight for the first signal. Default is 0.5 (50%).
    weight2 : float, optional
        Weight for the second signal. Default is 0.5 (50%).

    Returns
    -------
    np.ndarray
        Mixed audio as a float32 numpy array. Length equals the longer input.

    Raises
    ------
    ValueError
        If weights are invalid or arrays are incompatible.

    Example
    -------
    >>> import audiopy as ap
    >>> voice, sr = ap.load("voice.wav")
    >>> music, _ = ap.load("music.wav", sr=sr)
    >>> mixed = ap.mix(voice, music, weight1=0.7, weight2=0.3)
    """
    if weight1 < 0 or weight2 < 0:
        raise ValueError(
            f"Weights must be non-negative. Got weight1={weight1}, weight2={weight2}."
        )

    if y1.ndim != y2.ndim:
        raise ValueError(
            f"Audio arrays must have the same number of dimensions. "
            f"Got y1.ndim={y1.ndim}, y2.ndim={y2.ndim}."
        )

    try:
        # Pad shorter array to match longer
        if y1.ndim == 1:
            target_len = max(len(y1), len(y2))
            a1 = pad(y1, target_len)
            a2 = pad(y2, target_len)
        else:
            target_len = max(y1.shape[1], y2.shape[1])
            a1 = pad(y1, target_len)
            a2 = pad(y2, target_len)

        result = (weight1 * a1 + weight2 * a2).astype(np.float32)
        # Clip to prevent clipping distortion
        result = np.clip(result, -1.0, 1.0)
        return result
    except Exception as e:
        raise RuntimeError(
            f"Failed to mix audio signals.\nOriginal error: {e}"
        ) from e


def normalize(y: np.ndarray) -> np.ndarray:
    """
    Normalize audio to the range [-1.0, 1.0] by peak amplitude.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a numpy array.

    Returns
    -------
    np.ndarray
        Peak-normalized audio as float32. If the input is silent (all zeros),
        it is returned unchanged.

    Raises
    ------
    ValueError
        If the input is not a numpy array.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("quiet.wav")
    >>> y_norm = ap.normalize(y)
    >>> print(f"Peak amplitude: {np.max(np.abs(y_norm)):.4f}")  # ~1.0
    """
    if not isinstance(y, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(y).__name__}.")

    y_float = y.astype(np.float32)
    peak = np.max(np.abs(y_float))

    if peak == 0.0:
        return y_float  # Silent audio — return as-is

    return (y_float / peak).astype(np.float32)


def duration(y: np.ndarray, sr: int) -> float:
    """
    Calculate the duration of an audio array in seconds.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series. Shape: (samples,) or (channels, samples).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    float
        Duration of the audio in seconds.

    Raises
    ------
    ValueError
        If sr is not a positive integer.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> print(f"Duration: {ap.duration(y, sr):.2f} seconds")
    """
    if sr <= 0:
        raise ValueError(f"Sample rate must be a positive integer, got {sr}.")
    if not isinstance(y, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(y).__name__}.")

    n_samples = y.shape[-1] if y.ndim == 2 else y.shape[0]
    return float(n_samples) / float(sr)
