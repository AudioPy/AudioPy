"""
AI-powered sound component analysis using HuggingFace models
Analyzes audio files to identify and extract sound components
"""

import numpy as np
from typing import Dict, List, Any
import warnings

# Try to import required libraries
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers library not installed. Install with: pip install transformers torch")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not installed. Install with: pip install librosa")


class AudioAnalyzer:
    """
    Analyzes audio using HuggingFace models to identify sound components.
    
    Supports multiple models:
    - Audio classification (emotion, genre, etc.)
    - Audio tagging (sound event detection)
    - Audio to text (speech recognition, sound description)
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base", api_key: str = None):
        """
        Initialize the AudioAnalyzer with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
                - "facebook/wav2vec2-base" - Speech recognition
                - "openai/whisper-base" - Multilingual speech recognition
                - "superb/hubert-base-superb-ks" - Audio tagging (keyword spotting)
                - "facebook/wav2vec2-large-xlsr-53-english" - Audio representation
            api_key: HuggingFace API key (optional, for private models)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.pipeline = None
        self.task_type = self._infer_task_type(model_name)
        
        if HAS_TRANSFORMERS:
            try:
                self._initialize_pipeline()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize model: {e}")
        else:
            raise ImportError("transformers library is required. Install with: pip install transformers torch")
    
    def _infer_task_type(self, model_name: str) -> str:
        """Infer the task type from model name."""
        model_name_lower = model_name.lower()
        
        if "whisper" in model_name_lower or "wav2vec" in model_name_lower:
            return "automatic-speech-recognition"
        elif "tagging" in model_name_lower or "effnet" in model_name_lower:
            return "audio-classification"
        elif "emotion" in model_name_lower:
            return "audio-classification"
        else:
            return "audio-classification"
    
    def _initialize_pipeline(self):
        """Initialize the transformers pipeline with the specified model."""
        kwargs = {"model": self.model_name}
        if self.api_key:
            kwargs["use_auth_token"] = self.api_key
        
        self.pipeline = pipeline(task=self.task_type, **kwargs)
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio file and extract sound components.
        
        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.)
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        try:
            # Load audio file using our own soundfile I/O helper
            from .io import load
            samples, sr = load(audio_path)
            
            # Pass raw samples directly to the transformers pipeline
            results = self.pipeline({"raw": samples, "sampling_rate": sr}, top_k=None)
            
            return {
                "model": self.model_name,
                "task": self.task_type,
                "results": results,
                "components": self._extract_components(results)
            }
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")
    
    def _extract_components(self, results: Any) -> Dict[str, Any]:
        """
        Extract meaningful components from model results.
        
        Returns structured component information.
        """
        components = {
            "detected_sounds": [],
            "confidence_scores": [],
            "metadata": {}
        }
        
        # Handle different result formats
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    if "label" in item and "score" in item:
                        components["detected_sounds"].append(item["label"])
                        components["confidence_scores"].append(float(item["score"]))
                    elif "text" in item:
                        components["metadata"]["transcription"] = item["text"]
        elif isinstance(results, dict):
            if "text" in results:
                components["metadata"]["transcription"] = results["text"]
            if "tokens" in results:
                components["metadata"]["tokens"] = results["tokens"]
        
        return components
    
    def analyze_with_features(self, audio_path: str, samples: np.ndarray = None, 
                             sr: int = None) -> Dict[str, Any]:
        """
        Comprehensive analysis combining AI model with audio features.
        
        Args:
            audio_path: Path to audio file
            samples: Audio samples (numpy array). If provided, sr must also be provided
            sr: Sample rate. Required if samples is provided
            
        Returns:
            Dictionary with AI analysis and audio features
        """
        # Get AI model predictions
        ai_results = self.analyze_audio(audio_path)
        
        # Extract additional audio features if librosa is available
        features = {}
        if HAS_LIBROSA and (samples is not None and sr is not None):
            features = self._extract_audio_features(samples, sr)
        
        return {
            **ai_results,
            "audio_features": features
        }
    
    def _extract_audio_features(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract low-level audio features using librosa."""
        if not HAS_LIBROSA:
            return {}
        
        features = {}
        
        try:
            # Spectral features
            features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr)))
            features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=samples, sr=sr)))
            features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(samples)))
            
            # MFCC (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
            features["mfcc_std"] = np.std(mfcc, axis=1).tolist()
            
            # Tempogram (rhythm)
            onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
            features["onset_strength"] = float(np.mean(onset_env))
            
            # Chroma features (for music)
            chroma = librosa.feature.chroma_stft(y=samples, sr=sr)
            features["chroma_mean"] = np.mean(chroma, axis=1).tolist()
            
        except Exception as e:
            warnings.warn(f"Could not extract all audio features: {e}")
        
        return features


class SoundComponentExtractor:
    """
    High-level interface for extracting sound components from audio files.
    Provides simplified access to AI analysis with sensible defaults.
    """
    
    # Predefined model configurations
    MODELS = {
        "speech_recognition": "openai/whisper-base",
        "audio_tagging": "superb/hubert-base-superb-ks",  # Keyword spotting - audio event detection
        "emotion_detection": "facebook/wav2vec2-base",  # Speech emotion analysis
        "music_analysis": "facebook/wav2vec2-base",  # Music analysis
    }
    
    def __init__(self, analysis_type: str = "audio_tagging", api_key: str = None):
        """
        Initialize with a specific analysis type.
        
        Args:
            analysis_type: One of 'speech_recognition', 'audio_tagging', 
                          'emotion_detection', 'music_analysis', or a custom model name
            api_key: HuggingFace API key
        """
        model_name = self.MODELS.get(analysis_type, analysis_type)
        self.analyzer = AudioAnalyzer(model_name=model_name, api_key=api_key)
        self.analysis_type = analysis_type
    
    def extract_components(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract sound components from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with extracted components
        """
        return self.analyzer.analyze_audio(audio_path)
    
    def extract_components_with_features(self, audio_path: str, 
                                        samples: np.ndarray = None,
                                        sr: int = None) -> Dict[str, Any]:
        """
        Extract components with additional audio features.
        
        Args:
            audio_path: Path to audio file
            samples: Audio samples (optional)
            sr: Sample rate (optional)
            
        Returns:
            Dictionary with components and features
        """
        return self.analyzer.analyze_with_features(audio_path, samples, sr)