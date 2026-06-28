"""
audiopy.effects — Audio Effects Module
========================================
Digital signal processing effects: pitch shifting, time stretching, reverb,
echo, filtering, noise reduction, compression, and more.
Uses librosa, scipy, pedalboard, and noisereduce backends.
"""

import numpy as np

__all__ = [
    "change_pitch",
    "change_speed",
    "reverb",
    "echo",
    "low_pass_filter",
    "high_pass_filter",
    "bass_boost",
    "treble_boost",
    "distortion",
    "chorus",
    "noise_reduce",
    "noise_reduce_adaptive",
    "compress",
    "fade_in",
    "fade_out",
    "equalize",
]


def _ensure_mono_float32(y: np.ndarray) -> np.ndarray:
    """Internal: convert to mono float32."""
    if y.ndim == 2:
        return np.mean(y, axis=0).astype(np.float32)
    return y.astype(np.float32)


def change_pitch(y: np.ndarray, sr: int, steps: float) -> np.ndarray:
    """
    Shift the pitch of an audio signal by a number of semitones.

    Pitch shifting is performed without changing the playback speed (time-stretching
    is applied internally to compensate). Uses librosa's phase vocoder.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array (mono).
    sr : int
        Sample rate in Hz.
    steps : float
        Number of semitones to shift pitch. Positive values shift up,
        negative values shift down. E.g., ``steps=2`` shifts up by a
        major second, ``steps=-12`` shifts down one octave.

    Returns
    -------
    np.ndarray
        Pitch-shifted audio as float32. Same length as input.

    Raises
    ------
    ValueError
        If steps is not a finite number.
    RuntimeError
        If pitch shifting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("voice.wav")
    >>> y_high = ap.change_pitch(y, sr, steps=4)   # shift up 4 semitones
    >>> y_low  = ap.change_pitch(y, sr, steps=-3)  # shift down 3 semitones
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    if not np.isfinite(steps):
        raise ValueError(f"steps must be a finite number, got {steps}.")

    y_mono = _ensure_mono_float32(y)
    try:
        result = librosa.effects.pitch_shift(y_mono, sr=sr, n_steps=steps)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Pitch shifting by {steps} semitones failed.\nOriginal error: {e}"
        ) from e


def change_speed(y: np.ndarray, rate: float) -> np.ndarray:
    """
    Change the playback speed of an audio signal without altering pitch.

    Uses librosa's time-stretching (phase vocoder). A rate > 1 speeds up,
    a rate < 1 slows down. The output length changes proportionally.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    rate : float
        Speed multiplier. ``1.0`` = unchanged, ``2.0`` = 2× faster,
        ``0.5`` = half speed. Must be positive.

    Returns
    -------
    np.ndarray
        Time-stretched audio as float32.
        Output length ≈ len(y) / rate.

    Raises
    ------
    ValueError
        If rate is not a positive finite number.
    RuntimeError
        If time-stretching fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("speech.wav")
    >>> y_fast = ap.change_speed(y, rate=1.5)  # 1.5× speed
    >>> y_slow = ap.change_speed(y, rate=0.75) # 75% speed
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required. Install with: pip install librosa") from e

    if rate <= 0 or not np.isfinite(rate):
        raise ValueError(f"rate must be a positive finite number, got {rate}.")

    y_mono = _ensure_mono_float32(y)
    try:
        result = librosa.effects.time_stretch(y_mono, rate=rate)
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Speed change at rate={rate} failed.\nOriginal error: {e}"
        ) from e


def reverb(
    y: np.ndarray,
    sr: int,
    room_scale: float = 0.5,
    damping: float = 0.5,
    wet_level: float = 0.3,
) -> np.ndarray:
    """
    Add a realistic reverb effect using the Pedalboard library.

    Simulates the acoustic characteristics of a room or space by
    adding reflections and tail decay to the audio.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array (mono or stereo).
    sr : int
        Sample rate in Hz.
    room_scale : float, optional
        Size of the simulated room. Range [0.0, 1.0]. Default is 0.5.
    damping : float, optional
        High-frequency absorption of the reverberant tail. Range [0.0, 1.0].
        Higher values produce a warmer, more muffled reverb. Default is 0.5.
    wet_level : float, optional
        Mix level of the reverb signal. Range [0.0, 1.0]. Default is 0.3.
        0.0 = fully dry, 1.0 = fully wet.

    Returns
    -------
    np.ndarray
        Audio with reverb applied as float32.

    Raises
    ------
    ValueError
        If any parameter is out of the valid range.
    ImportError
        If pedalboard is not installed.
    RuntimeError
        If the effect fails to apply.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("voice.wav")
    >>> y_reverb = ap.reverb(y, sr, room_scale=0.8, wet_level=0.5)
    """
    try:
        from pedalboard import Pedalboard, Reverb
    except ImportError as e:
        raise ImportError(
            "pedalboard is required for reverb. Install with: pip install pedalboard"
        ) from e

    for name, val in [("room_scale", room_scale), ("damping", damping), ("wet_level", wet_level)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0.0, 1.0], got {val}.")

    y_float = y.astype(np.float32)
    try:
        board = Pedalboard([
            Reverb(
                room_size=room_scale,
                damping=damping,
                wet_level=wet_level,
                dry_level=1.0 - wet_level,
            )
        ])
        # pedalboard expects (channels, samples) for stereo or (1, samples) for mono
        if y_float.ndim == 1:
            audio_in = y_float.reshape(1, -1)
            result = board(audio_in, sr)
            return result.flatten().astype(np.float32)
        else:
            result = board(y_float, sr)
            return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Reverb effect failed.\nOriginal error: {e}") from e


def echo(y: np.ndarray, sr: int, delay: float = 0.3, decay: float = 0.4) -> np.ndarray:
    """
    Add an echo/delay effect to an audio signal.

    Produces a delayed copy of the audio mixed back with the original,
    creating a repeating echo effect.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    delay : float, optional
        Delay time in seconds. Default is 0.3 seconds.
    decay : float, optional
        Amplitude decay factor for each echo repetition. Range (0.0, 1.0).
        Lower values fade out faster. Default is 0.4.

    Returns
    -------
    np.ndarray
        Audio with echo applied as float32. Same length as input.

    Raises
    ------
    ValueError
        If delay is non-positive or decay is out of range.
    RuntimeError
        If the effect fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("voice.wav")
    >>> y_echo = ap.echo(y, sr, delay=0.4, decay=0.5)
    """
    if delay <= 0:
        raise ValueError(f"delay must be positive, got {delay}.")
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be in (0.0, 1.0), got {decay}.")

    y_float = _ensure_mono_float32(y)
    try:
        delay_samples = int(delay * sr)
        result = y_float.copy()
        delayed = np.zeros_like(y_float)

        # Apply multiple echo taps
        current_decay = decay
        current_delay = delay_samples
        while current_decay > 0.01 and current_delay < len(y_float):
            delayed[current_delay:] += y_float[: len(y_float) - current_delay] * current_decay
            current_delay += delay_samples
            current_decay *= decay

        result = result + delayed
        # Normalize to prevent clipping
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result /= peak
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Echo effect failed.\nOriginal error: {e}") from e


def low_pass_filter(y: np.ndarray, sr: int, cutoff: float = 3000.0) -> np.ndarray:
    """
    Apply a low-pass filter to remove frequencies above a cutoff.

    Uses a 5th-order Butterworth filter from scipy. Useful for removing
    high-frequency noise, creating a "muffled" or telephone effect.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    cutoff : float, optional
        Cutoff frequency in Hz. Frequencies above this are attenuated.
        Default is 3000 Hz.

    Returns
    -------
    np.ndarray
        Filtered audio as float32. Same shape as input.

    Raises
    ------
    ValueError
        If cutoff is not in the valid range (0, sr/2).
    RuntimeError
        If filtering fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> y_muffled = ap.low_pass_filter(y, sr, cutoff=1000)
    """
    try:
        from scipy.signal import butter, sosfilt
    except ImportError as e:
        raise ImportError("scipy is required for filtering. Install with: pip install scipy") from e

    nyquist = sr / 2.0
    if not (0 < cutoff < nyquist):
        raise ValueError(
            f"cutoff must be between 0 and Nyquist frequency ({nyquist:.0f} Hz). "
            f"Got {cutoff}."
        )

    y_float = y.astype(np.float32)
    try:
        sos = butter(5, cutoff / nyquist, btype="low", output="sos")
        if y_float.ndim == 1:
            return sosfilt(sos, y_float).astype(np.float32)
        else:
            return np.stack([
                sosfilt(sos, y_float[i]).astype(np.float32)
                for i in range(y_float.shape[0])
            ])
    except Exception as e:
        raise RuntimeError(f"Low-pass filter failed.\nOriginal error: {e}") from e


def high_pass_filter(y: np.ndarray, sr: int, cutoff: float = 300.0) -> np.ndarray:
    """
    Apply a high-pass filter to remove frequencies below a cutoff.

    Uses a 5th-order Butterworth filter from scipy. Useful for removing
    low-frequency rumble, hum, or DC offset.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    cutoff : float, optional
        Cutoff frequency in Hz. Frequencies below this are attenuated.
        Default is 300 Hz.

    Returns
    -------
    np.ndarray
        Filtered audio as float32. Same shape as input.

    Raises
    ------
    ValueError
        If cutoff is not in the valid range (0, sr/2).
    RuntimeError
        If filtering fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> y_clean = ap.high_pass_filter(y, sr, cutoff=100)
    """
    try:
        from scipy.signal import butter, sosfilt
    except ImportError as e:
        raise ImportError("scipy is required for filtering. Install with: pip install scipy") from e

    nyquist = sr / 2.0
    if not (0 < cutoff < nyquist):
        raise ValueError(
            f"cutoff must be between 0 and Nyquist frequency ({nyquist:.0f} Hz). "
            f"Got {cutoff}."
        )

    y_float = y.astype(np.float32)
    try:
        sos = butter(5, cutoff / nyquist, btype="high", output="sos")
        if y_float.ndim == 1:
            return sosfilt(sos, y_float).astype(np.float32)
        else:
            return np.stack([
                sosfilt(sos, y_float[i]).astype(np.float32)
                for i in range(y_float.shape[0])
            ])
    except Exception as e:
        raise RuntimeError(f"High-pass filter failed.\nOriginal error: {e}") from e


def bass_boost(y: np.ndarray, sr: int, gain_db: float = 6.0) -> np.ndarray:
    """
    Boost low-frequency (bass) content of an audio signal.

    Applies a low-shelf filter that amplifies frequencies below 250 Hz
    by the specified amount in dB.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    gain_db : float, optional
        Amount of bass boost in decibels. Positive values boost, negative
        values cut. Default is 6.0 dB.

    Returns
    -------
    np.ndarray
        Bass-boosted audio as float32.

    Raises
    ------
    RuntimeError
        If filtering fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("music.wav")
    >>> y_bass = ap.bass_boost(y, sr, gain_db=8)
    """
    try:
        from scipy.signal import butter, sosfilt
    except ImportError as e:
        raise ImportError("scipy is required. Install with: pip install scipy") from e

    y_float = y.astype(np.float32)
    try:
        nyquist = sr / 2.0
        cutoff = min(250.0, nyquist * 0.9)
        sos_lp = butter(2, cutoff / nyquist, btype="low", output="sos")
        gain_linear = 10 ** (gain_db / 20.0)

        if y_float.ndim == 1:
            low_content = sosfilt(sos_lp, y_float)
            result = y_float + low_content * (gain_linear - 1.0)
        else:
            result = np.stack([
                y_float[i] + sosfilt(sos_lp, y_float[i]) * (gain_linear - 1.0)
                for i in range(y_float.shape[0])
            ])

        # Normalize to prevent clipping
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result = result / peak
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Bass boost failed.\nOriginal error: {e}") from e


def treble_boost(y: np.ndarray, sr: int, gain_db: float = 6.0) -> np.ndarray:
    """
    Boost high-frequency (treble) content of an audio signal.

    Applies a high-shelf filter that amplifies frequencies above 4000 Hz
    by the specified amount in dB.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    gain_db : float, optional
        Amount of treble boost in decibels. Positive values boost, negative
        values cut. Default is 6.0 dB.

    Returns
    -------
    np.ndarray
        Treble-boosted audio as float32.

    Raises
    ------
    RuntimeError
        If filtering fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("music.wav")
    >>> y_bright = ap.treble_boost(y, sr, gain_db=4)
    """
    try:
        from scipy.signal import butter, sosfilt
    except ImportError as e:
        raise ImportError("scipy is required. Install with: pip install scipy") from e

    y_float = y.astype(np.float32)
    try:
        nyquist = sr / 2.0
        cutoff = min(4000.0, nyquist * 0.9)
        sos_hp = butter(2, cutoff / nyquist, btype="high", output="sos")
        gain_linear = 10 ** (gain_db / 20.0)

        if y_float.ndim == 1:
            high_content = sosfilt(sos_hp, y_float)
            result = y_float + high_content * (gain_linear - 1.0)
        else:
            result = np.stack([
                y_float[i] + sosfilt(sos_hp, y_float[i]) * (gain_linear - 1.0)
                for i in range(y_float.shape[0])
            ])

        peak = np.max(np.abs(result))
        if peak > 1.0:
            result = result / peak
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Treble boost failed.\nOriginal error: {e}") from e


def distortion(y: np.ndarray, drive: float = 0.5) -> np.ndarray:
    """
    Apply a soft-clipping distortion effect to an audio signal.

    Simulates analog tube saturation by applying a non-linear soft-clipping
    function. Higher drive values produce more aggressive distortion.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    drive : float, optional
        Drive (distortion intensity). Range [0.0, 1.0]. Default is 0.5.
        0.0 = no distortion, 1.0 = very heavy clipping.

    Returns
    -------
    np.ndarray
        Distorted audio as float32, normalized to [-1, 1].

    Raises
    ------
    ValueError
        If drive is not in [0.0, 1.0].

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("guitar.wav")
    >>> y_dist = ap.distortion(y, drive=0.8)
    """
    if not (0.0 <= drive <= 1.0):
        raise ValueError(f"drive must be in [0.0, 1.0], got {drive}.")

    y_float = y.astype(np.float32)
    try:
        # Map drive [0,1] to gain [1, 50]
        gain = 1.0 + drive * 49.0
        driven = y_float * gain
        # Soft clip using tanh
        result = np.tanh(driven)
        # Normalize
        peak = np.max(np.abs(result))
        if peak > 0:
            result /= peak
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Distortion effect failed.\nOriginal error: {e}") from e


def chorus(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply a chorus effect to an audio signal using Pedalboard.

    The chorus effect creates the impression of multiple instruments or
    voices playing simultaneously by adding slightly pitch-modulated
    and delayed copies of the signal.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Audio with chorus applied as float32.

    Raises
    ------
    ImportError
        If pedalboard is not installed.
    RuntimeError
        If the effect fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("guitar.wav")
    >>> y_chorus = ap.chorus(y, sr)
    """
    try:
        from pedalboard import Pedalboard, Chorus
    except ImportError as e:
        raise ImportError(
            "pedalboard is required for chorus. Install with: pip install pedalboard"
        ) from e

    y_float = y.astype(np.float32)
    try:
        board = Pedalboard([Chorus()])
        if y_float.ndim == 1:
            audio_in = y_float.reshape(1, -1)
            result = board(audio_in, sr)
            return result.flatten().astype(np.float32)
        else:
            result = board(y_float, sr)
            return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Chorus effect failed.\nOriginal error: {e}") from e


def noise_reduce(
    y: np.ndarray,
    sr: int,
    noise_clip: np.ndarray = None,
    prop_decrease: float = 1.0,
    stationary: bool = False,
) -> np.ndarray:
    """
    Reduce background noise from an audio signal.

    Uses the ``noisereduce`` library with a noise profile. If no noise sample
    is provided, the first 0.5 seconds of the audio are used as the noise
    reference (assumes the recording starts with ambient noise).

    Parameters
    ----------
    y : np.ndarray
        Audio time-series to denoise as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    noise_clip : np.ndarray, optional
        Reference noise sample. If ``None``, the first 0.5 seconds of ``y``
        are used as the noise profile.
    prop_decrease : float, optional
        Proportion of noise reduction to apply. Range [0.0, 1.0].
        1.0 = maximum noise reduction. Default is 1.0.
    stationary : bool, optional
        If ``True``, treats noise as stationary (constant, e.g., AC hum).
        If ``False``, uses adaptive non-stationary noise reduction.
        Default is ``False``.

    Returns
    -------
    np.ndarray
        Denoised audio as float32. Same shape as input.

    Raises
    ------
    ImportError
        If noisereduce is not installed.
    RuntimeError
        If noise reduction fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("noisy_speech.wav")
    >>> y_clean = ap.noise_reduce(y, sr)  # use first 0.5s as noise profile
    >>> # Or provide a dedicated noise sample:
    >>> noise, _ = ap.load("noise_only.wav", sr=sr)
    >>> y_clean = ap.noise_reduce(y, sr, noise_clip=noise, stationary=True)
    """
    try:
        import noisereduce as nr
    except ImportError as e:
        raise ImportError(
            "noisereduce is required for noise reduction. "
            "Install with: pip install noisereduce"
        ) from e

    y_float = _ensure_mono_float32(y)

    # Use first 0.5s as noise profile if not provided
    if noise_clip is None:
        noise_samples = int(0.5 * sr)
        noise_clip = y_float[:noise_samples] if len(y_float) > noise_samples else y_float

    noise_clip = noise_clip.astype(np.float32)

    try:
        result = nr.reduce_noise(
            y=y_float,
            y_noise=noise_clip,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=stationary,
        )
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Noise reduction failed.\nOriginal error: {e}"
        ) from e


def noise_reduce_adaptive(
    y: np.ndarray,
    sr: int,
    chunks: int = 4,
    prop_decrease: float = 0.9,
) -> np.ndarray:
    """
    Adaptive noise reduction that processes the audio in independent chunks.

    Splits the audio into segments, reduces noise in each segment separately
    using a local noise profile, and concatenates the results. Better than
    ``noise_reduce`` for audio with changing background noise.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    chunks : int, optional
        Number of equal-length segments to split the audio into. Default is 4.
    prop_decrease : float, optional
        Noise reduction strength per chunk. Range [0.0, 1.0]. Default is 0.9.

    Returns
    -------
    np.ndarray
        Adaptively denoised audio as float32. Same length as input.

    Raises
    ------
    ValueError
        If chunks is not a positive integer.
    ImportError
        If noisereduce is not installed.
    RuntimeError
        If adaptive noise reduction fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("outdoor_recording.wav")
    >>> y_clean = ap.noise_reduce_adaptive(y, sr, chunks=8, prop_decrease=0.85)
    """
    try:
        import noisereduce as nr
    except ImportError as e:
        raise ImportError(
            "noisereduce is required. Install with: pip install noisereduce"
        ) from e

    if chunks <= 0:
        raise ValueError(f"chunks must be a positive integer, got {chunks}.")

    y_float = _ensure_mono_float32(y)

    try:
        chunk_size = len(y_float) // chunks
        if chunk_size == 0:
            return noise_reduce(y_float, sr, prop_decrease=prop_decrease)

        denoised_chunks = []
        for i in range(chunks):
            start = i * chunk_size
            end = start + chunk_size if i < chunks - 1 else len(y_float)
            chunk = y_float[start:end]

            # Use the beginning of each chunk as noise profile
            noise_len = max(1, len(chunk) // 10)
            noise_sample = chunk[:noise_len]

            try:
                denoised = nr.reduce_noise(
                    y=chunk,
                    y_noise=noise_sample,
                    sr=sr,
                    prop_decrease=prop_decrease,
                    stationary=False,
                )
            except Exception:
                denoised = chunk  # Fallback: return original chunk

            denoised_chunks.append(denoised.astype(np.float32))

        return np.concatenate(denoised_chunks).astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Adaptive noise reduction failed.\nOriginal error: {e}"
        ) from e


def compress(
    y: np.ndarray,
    threshold: float = -20.0,
    ratio: float = 4.0,
) -> np.ndarray:
    """
    Apply dynamic range compression to an audio signal.

    Reduces the dynamic range by attenuating audio that exceeds the threshold
    by a specified ratio. Useful for evening out loud and quiet passages.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    threshold : float, optional
        Threshold in dBFS above which compression is applied. Default is -20 dBFS.
    ratio : float, optional
        Compression ratio (N:1). For example, 4.0 means that for every 4 dB
        the signal exceeds the threshold, only 1 dB passes. Must be >= 1.0.
        Default is 4.0.

    Returns
    -------
    np.ndarray
        Compressed audio as float32. Normalized to prevent clipping.

    Raises
    ------
    ValueError
        If ratio is less than 1.0.
    RuntimeError
        If compression fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("dynamic_audio.wav")
    >>> y_compressed = ap.compress(y, threshold=-18, ratio=6.0)
    """
    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1.0, got {ratio}.")

    y_float = y.astype(np.float32)
    try:
        # Convert threshold from dBFS to linear amplitude
        thresh_linear = 10 ** (threshold / 20.0)

        def compress_sample(sample):
            abs_val = abs(sample)
            if abs_val <= thresh_linear:
                return sample
            # Amount above threshold in linear
            excess = abs_val - thresh_linear
            # Apply ratio
            compressed_excess = excess / ratio
            compressed_abs = thresh_linear + compressed_excess
            return np.sign(sample) * compressed_abs

        result = np.vectorize(compress_sample)(y_float)

        # Normalize to prevent clipping
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result /= peak
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Compression failed.\nOriginal error: {e}") from e


def fade_in(y: np.ndarray, sr: int, duration: float = 0.5) -> np.ndarray:
    """
    Apply a linear fade-in to the beginning of an audio signal.

    The audio starts at zero amplitude and linearly ramps up to full
    amplitude over the specified duration.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    duration : float, optional
        Length of the fade-in in seconds. Default is 0.5 seconds.
        Must not exceed the total audio duration.

    Returns
    -------
    np.ndarray
        Audio with fade-in applied as float32. Same shape as input.

    Raises
    ------
    ValueError
        If duration is non-positive.
    RuntimeError
        If the effect fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> y_faded = ap.fade_in(y, sr, duration=1.0)
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}.")

    y_float = y.astype(np.float32)
    try:
        fade_samples = min(int(duration * sr), y_float.shape[-1])
        envelope = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        result = y_float.copy()
        if result.ndim == 1:
            result[:fade_samples] *= envelope
        else:
            result[:, :fade_samples] *= envelope
        return result
    except Exception as e:
        raise RuntimeError(f"Fade-in failed.\nOriginal error: {e}") from e


def fade_out(y: np.ndarray, sr: int, duration: float = 0.5) -> np.ndarray:
    """
    Apply a linear fade-out to the end of an audio signal.

    The audio linearly decreases from full amplitude to zero over the
    specified duration at the end of the signal.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    duration : float, optional
        Length of the fade-out in seconds. Default is 0.5 seconds.
        Must not exceed the total audio duration.

    Returns
    -------
    np.ndarray
        Audio with fade-out applied as float32. Same shape as input.

    Raises
    ------
    ValueError
        If duration is non-positive.
    RuntimeError
        If the effect fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> y_faded = ap.fade_out(y, sr, duration=2.0)
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}.")

    y_float = y.astype(np.float32)
    try:
        fade_samples = min(int(duration * sr), y_float.shape[-1])
        envelope = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        result = y_float.copy()
        if result.ndim == 1:
            result[-fade_samples:] *= envelope
        else:
            result[:, -fade_samples:] *= envelope
        return result
    except Exception as e:
        raise RuntimeError(f"Fade-out failed.\nOriginal error: {e}") from e


def equalize(y: np.ndarray, sr: int, bands: dict) -> np.ndarray:
    """
    Apply a 3-band parametric equalizer to an audio signal.

    Adjusts the gain of low, mid, and high frequency ranges independently.
    Uses Butterworth filters from scipy to implement each band.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    bands : dict
        EQ gains in dB for each band. Supported keys:

        - ``"low"``  : gain in dB for frequencies below 300 Hz
        - ``"mid"``  : gain in dB for frequencies between 300 Hz and 3000 Hz
        - ``"high"`` : gain in dB for frequencies above 3000 Hz

        Example: ``{"low": 3, "mid": 0, "high": -2}``
        Positive values boost, negative values cut. Omitted bands default to 0 dB.

    Returns
    -------
    np.ndarray
        Equalized audio as float32.

    Raises
    ------
    ValueError
        If bands dict contains invalid keys or gain values are not finite.
    RuntimeError
        If equalization fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("music.wav")
    >>> y_eq = ap.equalize(y, sr, bands={"low": 4, "mid": -2, "high": 3})
    """
    try:
        from scipy.signal import butter, sosfilt
    except ImportError as e:
        raise ImportError("scipy is required. Install with: pip install scipy") from e

    valid_keys = {"low", "mid", "high"}
    invalid_keys = set(bands.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(
            f"Invalid band keys: {invalid_keys}. "
            f"Valid keys are: {valid_keys}"
        )

    low_gain_db = bands.get("low", 0.0)
    mid_gain_db = bands.get("mid", 0.0)
    high_gain_db = bands.get("high", 0.0)

    for name, val in [("low", low_gain_db), ("mid", mid_gain_db), ("high", high_gain_db)]:
        if not np.isfinite(val):
            raise ValueError(f"'{name}' gain must be a finite number, got {val}.")

    y_float = y.astype(np.float32)
    nyquist = sr / 2.0

    try:
        # Band frequency boundaries
        low_cutoff = min(300.0, nyquist * 0.9)
        high_cutoff = min(3000.0, nyquist * 0.9)

        # Low band: LPF below 300 Hz
        sos_low = butter(2, low_cutoff / nyquist, btype="low", output="sos")
        # High band: HPF above 3000 Hz
        sos_high = butter(2, high_cutoff / nyquist, btype="high", output="sos")
        # Mid band: BPF between 300 Hz and 3000 Hz
        if high_cutoff > low_cutoff:
            sos_mid = butter(2, [low_cutoff / nyquist, high_cutoff / nyquist],
                             btype="band", output="sos")
        else:
            sos_mid = None

        low_gain = 10 ** (low_gain_db / 20.0)
        mid_gain = 10 ** (mid_gain_db / 20.0)
        high_gain = 10 ** (high_gain_db / 20.0)

        def apply_eq(channel):
            low_band = sosfilt(sos_low, channel) * low_gain
            high_band = sosfilt(sos_high, channel) * high_gain
            if sos_mid is not None:
                mid_band = sosfilt(sos_mid, channel) * mid_gain
            else:
                mid_band = np.zeros_like(channel)
            return (low_band + mid_band + high_band).astype(np.float32)

        if y_float.ndim == 1:
            result = apply_eq(y_float)
        else:
            result = np.stack([apply_eq(y_float[i]) for i in range(y_float.shape[0])])

        # Normalize to prevent clipping
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result /= peak
        return result.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Equalization failed.\nOriginal error: {e}") from e
