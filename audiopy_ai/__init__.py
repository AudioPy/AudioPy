"""
audiopy_ai - Audio editing and effects library with AI-powered analysis

Modules:
    io: Audio file I/O (load, save)
    edit: Audio editing functions (trim, fade, normalize, etc.)
    effects: Audio effects (filters, reverb, time effects)
    ai_analysis: AI-powered sound component analysis using HuggingFace models
"""

import os
import subprocess
from pathlib import Path

def _add_ffmpeg_to_path():
    try:
        # Check if ffmpeg is already accessible
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # Fallback: check workspace directories
        current_file_dir = Path(__file__).parent
        repo_root = current_file_dir.parent
        parent_dir = repo_root.parent
        
        for directory in [repo_root, parent_dir]:
            ffmpeg_path = directory / "ffmpeg.exe"
            if ffmpeg_path.exists():
                os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
                return
                
        # Fallback: check common paths and Wondershare
        common_paths = [
            Path("C:/Program Files/Wondershare/Wondershare UniConverter 14 for Windows"),
            Path("C:/Program Files/Wondershare/Wondershare UniConverter 14 for Windows/DownloadRes"),
            Path("C:/Program Files/ffmpeg/bin"),
            Path("C:/ffmpeg/bin"),
        ]
        
        user_profile = os.environ.get('USERPROFILE')
        if user_profile:
            common_paths.extend([
                Path(user_profile) / "scoop" / "shims",
                Path(user_profile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin"
            ])
            
        for directory in common_paths:
            ffmpeg_path = directory / "ffmpeg.exe"
            if ffmpeg_path.exists():
                os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
                return
                    
        # Fallback: check winget folder
        local_appdata = os.environ.get('LOCALAPPDATA')
        if local_appdata:
            winget_packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
            if winget_packages.exists():
                ffmpeg_bins = list(winget_packages.glob("**/ffmpeg.exe"))
                if ffmpeg_bins:
                    ffmpeg_dir = str(ffmpeg_bins[0].parent)
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

_add_ffmpeg_to_path()

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

__version__ = "0.1.5"
__author__ = "audiopy_ai Team"

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
