"""
AudioPy - Audio editing and effects library with AI-powered analysis

Modules:
    io: Audio file I/O (load, save)
    edit: Audio editing functions (trim, fade, normalize, etc.)
    effects: Audio effects (filters, reverb, time effects)
    ai_analysis: AI-powered sound component analysis using HuggingFace models
"""

# Core I/O
from .io import load, save

# Audio editing
from .edit import (
    trim,
    concat,
    apply_gain,
    rms_db,
    normalize,
    fade,
    reverse
)

# AI Analysis
try:
    from .ai_analysis import AudioAnalyzer, SoundComponentExtractor
except ImportError:
    AudioAnalyzer = None
    SoundComponentExtractor = None

__version__ = "0.1.0"
__author__ = "AudioPy Team"

__all__ = [
    # I/O
    'load',
    'save',
    
    # Editing
    'trim',
    'concat',
    'apply_gain',
    'rms_db',
    'normalize',
    'fade',
    'reverse',
    
    # AI Analysis
    'AudioAnalyzer',
    'SoundComponentExtractor',
]
