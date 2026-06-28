"""
audiopy — Python Audio ML Library
====================================
A comprehensive audio processing and machine learning library for Python,
inspired by OpenCV's design philosophy: simple imports, powerful capabilities.

Usage
-----
    import audiopy as ap

    # Load audio
    y, sr = ap.load("audio.wav")

    # Basic processing
    y_mono  = ap.mono(y)
    y_clean = ap.noise_reduce(y, sr)
    y_norm  = ap.normalize(y_clean)

    # Features
    mfccs   = ap.mfcc(y, sr)

    # Visualization
    fig = ap.plot_waveform(y, sr)

    # AI models
    text    = ap.transcribe(y, sr)
    emotion = ap.detect_emotion(y, sr)
    stems   = ap.separate_vocals(y, sr)

    # Save
    ap.save("output.wav", y_clean, sr)

Submodules
----------
- ap.io       : Load, save, record, stream audio
- ap.core     : Resample, trim, pad, mix, normalize
- ap.features : MFCC, spectrogram, chroma, pitch, tempo
- ap.effects  : Reverb, echo, EQ, noise reduction, pitch shift
- ap.viz      : Waveform, spectrogram, MFCC, beat visualization
- ap.utils    : dB conversion, stats, format detection
- ap.models   : Transcription, classification, emotion, TTS, separation

Version
-------
0.1.0
"""

__version__ = "0.1.0"
__author__ = "audiopy contributors"
__license__ = "MIT"

# ─────────────────────────────────────────────────────────────────────────────
# I/O — Load, Save, Record
# ─────────────────────────────────────────────────────────────────────────────
from .io import (
    load,
    save,
    record,
    from_url,
    info,
)

# ─────────────────────────────────────────────────────────────────────────────
# Core Processing
# ─────────────────────────────────────────────────────────────────────────────
from .core import (
    resample,
    mono,
    stereo,
    trim,
    pad,
    split_on_silence,
    concatenate,
    mix,
    normalize,
    duration,
)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
from .features import (
    mfcc,
    mel_spectrogram,
    spectrogram,
    chroma,
    zero_crossing_rate,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    rms_energy,
    tempo,
    pitch,
    beat_frames,
    extract_all,
)

# ─────────────────────────────────────────────────────────────────────────────
# Audio Effects
# ─────────────────────────────────────────────────────────────────────────────
from .effects import (
    change_pitch,
    change_speed,
    reverb,
    echo,
    low_pass_filter,
    high_pass_filter,
    bass_boost,
    treble_boost,
    distortion,
    chorus,
    noise_reduce,
    noise_reduce_adaptive,
    compress,
    fade_in,
    fade_out,
    equalize,
)

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
from .viz import (
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_chroma,
    plot_pitch,
    plot_beats,
    compare_waveforms,
    save_plot,
)

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
from . import utils

# ─────────────────────────────────────────────────────────────────────────────
# AI Models — Lazy-loaded, only downloaded on first use
# ─────────────────────────────────────────────────────────────────────────────
from . import models

# Transcription (Speech-to-Text)
from .models.transcribe import (
    transcribe,
    transcribe_with_timestamps,
    translate_to_english,
)

# Audio Classification
from .models.classifier import (
    classify,
    is_speech,
    classify_music_genre,
)

# Emotion Recognition
from .models.emotion import (
    detect_emotion,
    dominant_emotion,
)

# Source Separation
from .models.separation import (
    separate_vocals,
    separate_instruments,
)

# Text-to-Speech
from .models.tts import (
    speak,
    speak_advanced,
)

# Speech Enhancement
from .models.enhancement import (
    denoise_deep,
    enhance_speech,
    remove_silence,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Package metadata
    "__version__",
    "__author__",

    # I/O
    "load",
    "save",
    "record",
    "from_url",
    "info",

    # Core
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

    # Features
    "mfcc",
    "mel_spectrogram",
    "spectrogram",
    "chroma",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "rms_energy",
    "tempo",
    "pitch",
    "beat_frames",
    "extract_all",

    # Effects
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

    # Visualization
    "plot_waveform",
    "plot_spectrogram",
    "plot_mel_spectrogram",
    "plot_mfcc",
    "plot_chroma",
    "plot_pitch",
    "plot_beats",
    "compare_waveforms",
    "save_plot",

    # Utilities
    "utils",

    # Models namespace
    "models",

    # Transcription
    "transcribe",
    "transcribe_with_timestamps",
    "translate_to_english",

    # Classification
    "classify",
    "is_speech",
    "classify_music_genre",

    # Emotion
    "detect_emotion",
    "dominant_emotion",

    # Separation
    "separate_vocals",
    "separate_instruments",

    # TTS
    "speak",
    "speak_advanced",

    # Enhancement
    "denoise_deep",
    "enhance_speech",
    "remove_silence",
]
