"""
audiopy.viz — Audio Visualization Module
==========================================
Clean, publication-ready plots for waveforms, spectrograms, features,
and comparisons. All functions return matplotlib Figure objects — they do
NOT call plt.show() so users can choose to display, save, or embed them.
"""

import numpy as np

__all__ = [
    "plot_waveform",
    "plot_spectrogram",
    "plot_mel_spectrogram",
    "plot_mfcc",
    "plot_chroma",
    "plot_pitch",
    "plot_beats",
    "compare_waveforms",
    "save_plot",
]

# ── Shared style defaults ──────────────────────────────────────────────────
_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#4a4a8a",
    "axes.labelcolor": "#e0e0ff",
    "xtick.color": "#a0a0cc",
    "ytick.color": "#a0a0cc",
    "text.color": "#e0e0ff",
    "grid.color": "#2a2a5a",
    "grid.alpha": 0.4,
}


def _apply_style():
    """Apply the audiopy dark theme to matplotlib."""
    import matplotlib.pyplot as plt
    for key, val in _STYLE.items():
        plt.rcParams[key] = val


def plot_waveform(y: np.ndarray, sr: int, title: str = "Waveform"):
    """
    Plot the audio waveform (amplitude vs. time).

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz. Used to compute the time axis.
    title : str, optional
        Title of the plot. Default is ``"Waveform"``.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object. Call ``fig.show()`` or
        ``ap.save_plot(fig, "waveform.png")`` to display or save.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> fig = ap.plot_waveform(y, sr, title="My Audio")
    >>> fig.show()
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        ) from e

    _apply_style()

    y_plot = y if y.ndim == 1 else np.mean(y, axis=0)
    times = np.linspace(0, len(y_plot) / sr, num=len(y_plot))

    try:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(times, y_plot, color="#7b68ee", linewidth=0.7, alpha=0.9)
        ax.fill_between(times, y_plot, alpha=0.15, color="#7b68ee")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_xlim(0, times[-1])
        ax.axhline(0, color="#4a4a8a", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Waveform plot failed.\nOriginal error: {e}") from e


def plot_spectrogram(y: np.ndarray, sr: int, title: str = "Spectrogram"):
    """
    Plot the magnitude spectrogram (dB scale) of an audio signal.

    Uses the Short-Time Fourier Transform (STFT) to compute the spectrogram.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    title : str, optional
        Title of the plot. Default is ``"Spectrogram"``.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.

    Raises
    ------
    ImportError
        If matplotlib or librosa is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> fig = ap.plot_spectrogram(y, sr)
    >>> ap.save_plot(fig, "spectrogram.png")
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError as e:
        raise ImportError(
            "matplotlib and librosa are required. "
            "Install with: pip install matplotlib librosa"
        ) from e

    _apply_style()

    y_mono = y if y.ndim == 1 else np.mean(y, axis=0)

    try:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_mono.astype(np.float32))), ref=np.max)
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Spectrogram plot failed.\nOriginal error: {e}") from e


def plot_mel_spectrogram(y: np.ndarray, sr: int, title: str = "Mel Spectrogram"):
    """
    Plot the Mel-scale power spectrogram of an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    title : str, optional
        Title of the plot. Default is ``"Mel Spectrogram"``.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.

    Raises
    ------
    ImportError
        If matplotlib or librosa is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> fig = ap.plot_mel_spectrogram(y, sr)
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError as e:
        raise ImportError(
            "matplotlib and librosa are required. "
            "Install with: pip install matplotlib librosa"
        ) from e

    _apply_style()

    y_mono = y if y.ndim == 1 else np.mean(y, axis=0)

    try:
        S = librosa.feature.melspectrogram(y=y_mono.astype(np.float32), sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis"
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Power (dB)")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Mel Frequency", fontsize=11)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Mel spectrogram plot failed.\nOriginal error: {e}") from e


def plot_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13):
    """
    Plot the Mel-Frequency Cepstral Coefficients (MFCCs) as a heatmap.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.
    n_mfcc : int, optional
        Number of MFCCs to compute and display. Default is 13.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.

    Raises
    ------
    ImportError
        If matplotlib or librosa is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("speech.wav")
    >>> fig = ap.plot_mfcc(y, sr, n_mfcc=20)
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError as e:
        raise ImportError(
            "matplotlib and librosa are required. "
            "Install with: pip install matplotlib librosa"
        ) from e

    _apply_style()

    y_mono = y if y.ndim == 1 else np.mean(y, axis=0)

    try:
        mfccs = librosa.feature.mfcc(y=y_mono.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            mfccs, sr=sr, x_axis="time", ax=ax, cmap="coolwarm"
        )
        fig.colorbar(img, ax=ax, label="MFCC Coefficient")
        ax.set_title(f"MFCCs ({n_mfcc} coefficients)", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("MFCC Index", fontsize=11)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"MFCC plot failed.\nOriginal error: {e}") from e


def plot_chroma(y: np.ndarray, sr: int):
    """
    Plot the chromagram (pitch class intensities over time).

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.

    Raises
    ------
    ImportError
        If matplotlib or librosa is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("song.wav")
    >>> fig = ap.plot_chroma(y, sr)
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError as e:
        raise ImportError(
            "matplotlib and librosa are required. "
            "Install with: pip install matplotlib librosa"
        ) from e

    _apply_style()

    y_mono = y if y.ndim == 1 else np.mean(y, axis=0)

    try:
        chroma_feat = librosa.feature.chroma_stft(y=y_mono.astype(np.float32), sr=sr)
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            chroma_feat, sr=sr, x_axis="time", y_axis="chroma", ax=ax, cmap="plasma"
        )
        fig.colorbar(img, ax=ax, label="Intensity")
        ax.set_title("Chromagram", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Pitch Class", fontsize=11)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Chroma plot failed.\nOriginal error: {e}") from e


def plot_pitch(y: np.ndarray, sr: int):
    """
    Plot the fundamental frequency (pitch) over time using pYIN estimation.

    Voiced frames (where pitch is detected) are plotted in color;
    unvoiced frames (np.nan) are skipped.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.

    Raises
    ------
    ImportError
        If matplotlib or librosa is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("singing.wav")
    >>> fig = ap.plot_pitch(y, sr)
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
    except ImportError as e:
        raise ImportError(
            "matplotlib and librosa are required. "
            "Install with: pip install matplotlib librosa"
        ) from e

    _apply_style()

    y_mono = y if y.ndim == 1 else np.mean(y, axis=0)

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_mono.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        times = librosa.times_like(f0, sr=sr)

        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot waveform as background
        waveform_times = np.linspace(0, len(y_mono) / sr, len(y_mono))
        ax2 = ax.twinx()
        ax2.plot(waveform_times, y_mono, color="#4a4a8a", linewidth=0.4, alpha=0.4)
        ax2.set_ylabel("Amplitude", color="#4a4a8a", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#4a4a8a")

        # Plot pitch (only voiced frames)
        voiced_times = times[voiced_flag]
        voiced_f0 = f0[voiced_flag]
        ax.scatter(voiced_times, voiced_f0, s=4, color="#00d4ff", alpha=0.9,
                   label="Fundamental Frequency (F0)", zorder=5)

        ax.set_title("Pitch (F0) Over Time", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)", fontsize=11, color="#00d4ff")
        ax.tick_params(axis="y", labelcolor="#00d4ff")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Pitch plot failed.\nOriginal error: {e}") from e


def plot_beats(y: np.ndarray, sr: int):
    """
    Plot detected beat positions overlaid on the audio waveform.

    Displays the waveform with vertical markers at detected beat positions
    and shows the estimated tempo in the title.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series as a float32 numpy array.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.

    Raises
    ------
    ImportError
        If matplotlib or librosa is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("music.wav")
    >>> fig = ap.plot_beats(y, sr)
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
    except ImportError as e:
        raise ImportError(
            "matplotlib and librosa are required. "
            "Install with: pip install matplotlib librosa"
        ) from e

    _apply_style()

    y_mono = y if y.ndim == 1 else np.mean(y, axis=0)

    try:
        tempo_val, beats = librosa.beat.beat_track(y=y_mono.astype(np.float32), sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        waveform_times = np.linspace(0, len(y_mono) / sr, len(y_mono))

        # Handle array return in newer librosa versions
        if hasattr(tempo_val, '__len__'):
            tempo_val = float(tempo_val[0])
        else:
            tempo_val = float(tempo_val)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(waveform_times, y_mono, color="#7b68ee", linewidth=0.6, alpha=0.85)
        for bt in beat_times:
            ax.axvline(x=bt, color="#ff6b6b", linewidth=1.0, alpha=0.8)

        ax.set_title(f"Beat Detection  |  Tempo: {tempo_val:.1f} BPM",
                     fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_xlim(0, waveform_times[-1])
        ax.grid(True, axis="x", alpha=0.2)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#7b68ee", linewidth=1.5, label="Waveform"),
            Line2D([0], [0], color="#ff6b6b", linewidth=1.5, label="Beat"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Beat plot failed.\nOriginal error: {e}") from e


def compare_waveforms(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    labels: tuple = ("Original", "Processed"),
):
    """
    Display two audio waveforms side-by-side for comparison.

    Useful for visually evaluating the effect of an audio processing step
    such as noise reduction, equalization, or pitch shifting.

    Parameters
    ----------
    y1 : np.ndarray
        First audio signal (e.g., original).
    y2 : np.ndarray
        Second audio signal (e.g., processed).
    sr : int
        Sample rate in Hz.
    labels : tuple of str, optional
        Labels for the two waveforms. Default is ``("Original", "Processed")``.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing two subplot axes.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    RuntimeError
        If plotting fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("noisy.wav")
    >>> y_clean = ap.noise_reduce(y, sr)
    >>> fig = ap.compare_waveforms(y, y_clean, sr, labels=("Noisy", "Denoised"))
    >>> fig.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    _apply_style()

    y1_plot = y1 if y1.ndim == 1 else np.mean(y1, axis=0)
    y2_plot = y2 if y2.ndim == 1 else np.mean(y2, axis=0)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)
        colors = ["#7b68ee", "#00d4ff"]

        for ax, y_plot, label, color in zip(axes, [y1_plot, y2_plot], labels, colors):
            times = np.linspace(0, len(y_plot) / sr, len(y_plot))
            ax.plot(times, y_plot, color=color, linewidth=0.7, alpha=0.9)
            ax.fill_between(times, y_plot, alpha=0.12, color=color)
            ax.set_title(label, fontsize=13, fontweight="bold", pad=8)
            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Amplitude", fontsize=11)
            ax.set_xlim(0, times[-1])
            ax.axhline(0, color="#4a4a8a", linewidth=0.5)
            ax.grid(True, axis="x", alpha=0.2)

        fig.suptitle("Waveform Comparison", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Waveform comparison plot failed.\nOriginal error: {e}") from e


def save_plot(fig, path: str) -> None:
    """
    Save a matplotlib Figure to a file on disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save (returned by any audiopy viz function).
    path : str
        Output file path including extension (e.g., ``"plot.png"``,
        ``"spectrogram.pdf"``). The format is inferred from the extension.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If fig is not a valid matplotlib Figure.
    RuntimeError
        If saving fails.

    Example
    -------
    >>> import audiopy as ap
    >>> y, sr = ap.load("audio.wav")
    >>> fig = ap.plot_waveform(y, sr)
    >>> ap.save_plot(fig, "waveform.png")
    """
    import os

    try:
        import matplotlib
        from matplotlib.figure import Figure
    except ImportError as e:
        raise ImportError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    if not isinstance(fig, Figure):
        raise ValueError(
            f"Expected a matplotlib Figure object, got {type(fig).__name__}."
        )

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    try:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[audiopy] Plot saved to: {path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to save plot to '{path}'.\nOriginal error: {e}"
        ) from e
