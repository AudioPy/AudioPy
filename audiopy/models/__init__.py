"""
audiopy.models — AI Model Subpackage
======================================
Exposes all deep learning model functions under the ``ap.models`` namespace.
All models are lazy-loaded: weights are only downloaded on first call.
"""

from . import transcribe
from . import classifier
from . import emotion
from . import separation
from . import tts
from . import enhancement

from .transcribe import (
    transcribe as transcribe_audio,
    transcribe_with_timestamps,
    translate_to_english,
)
from .classifier import (
    classify,
    is_speech,
    classify_music_genre,
)
from .emotion import (
    detect_emotion,
    dominant_emotion,
)
from .separation import (
    separate_vocals,
    separate_instruments,
)
from .tts import (
    speak,
    speak_advanced,
)
from .enhancement import (
    denoise_deep,
    enhance_speech,
    remove_silence,
)

__all__ = [
    # Submodules
    "transcribe",
    "classifier",
    "emotion",
    "separation",
    "tts",
    "enhancement",
    # Transcription
    "transcribe_audio",
    "transcribe_with_timestamps",
    "translate_to_english",
    # Classification
    "classify",
    "is_speech",
    "classify_music_genre",
    # Emotion
    "detect_emotion",
    "dominant_emotion",
    # Source separation
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
