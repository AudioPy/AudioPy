"""
setup.py — audiopy package installer
"""

from setuptools import setup, find_packages
import os


# Read long description from README
def read_file(filename):
    here = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(here, filename)
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    return ""


long_description = read_file("README.md")

setup(
    name="audiopy",
    version="0.1.0",
    author="audiopy contributors",
    author_email="audiopy@example.com",
    description=(
        "A comprehensive Python audio ML library — like OpenCV but for audio. "
        "Supports loading, processing, feature extraction, effects, visualization, "
        "and AI-powered transcription, emotion detection, source separation, and TTS."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/audiopy",
    project_urls={
        "Bug Tracker": "https://github.com/example/audiopy/issues",
        "Documentation": "https://audiopy.readthedocs.io",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        # Core audio I/O
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "sounddevice>=0.4.6",
        # Numerical
        "numpy>=1.23.0",
        # Visualization
        "matplotlib>=3.7.0",
        # Signal processing
        "scipy>=1.10.0",
        # Deep learning (core models)
        "transformers>=4.38.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        # Audio effects
        "noisereduce>=3.0.0",
        "pedalboard>=0.8.0",
        # Text-to-Speech
        "gTTS>=2.3.0",
        # Networking
        "requests>=2.28.0",
    ],
    extras_require={
        # Optional: Source separation (heavy dependencies)
        "separation": [
            "demucs>=4.0.0",
            "spleeter>=2.3.0",
        ],
        # Optional: WebRTC-based silence removal
        "vad": [
            "webrtcvad>=2.0.10",
        ],
        # Optional: Datasets for SpeechT5 speaker embeddings
        "advanced_tts": [
            "datasets>=2.14.0",
        ],
        # Optional: DeepFilterNet for deep denoising
        "deepfilter": [
            "deepfilternet>=0.5.0",
        ],
        # All optional extras combined
        "full": [
            "demucs>=4.0.0",
            "spleeter>=2.3.0",
            "webrtcvad>=2.0.10",
            "datasets>=2.14.0",
            "deepfilternet>=0.5.0",
        ],
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={},
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "audio", "speech", "music", "signal-processing", "machine-learning",
        "deep-learning", "speech-recognition", "emotion-detection",
        "source-separation", "text-to-speech", "whisper", "wav2vec2",
        "librosa", "pytorch", "huggingface", "transcription",
    ],
)
