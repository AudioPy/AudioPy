"""
Example 1: Basic Audio Component Analysis
Demonstrates how to use AudioAnalyzer for sound component extraction
"""

from audiopy_ai.ai_analysis import AudioAnalyzer, SoundComponentExtractor
from audiopy_ai.io import load, save
import json


def example_1_basic_analysis():
    """Basic usage: Analyze an audio file for sound components"""
    
    # Initialize analyzer with default audio tagging model
    analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
    
    # Analyze an audio file
    results = analyzer.extract_components("path/to/your/audio.wav")
    
    # Print detected sound components
    print("Detected Sound Components:")
    print(json.dumps(results, indent=2, default=str))


def example_2_with_api_key():
    """Using private HuggingFace models with API key"""
    
    # Initialize with your HuggingFace API key
    analyzer = SoundComponentExtractor(
        analysis_type="audio_tagging",
        api_key="hf_your_api_key_here"
    )
    
    # Analyze audio
    results = analyzer.extract_components("path/to/your/audio.wav")
    
    # Access detected components
    components = results['components']['detected_sounds']
    scores = results['components']['confidence_scores']
    
    print("Detected sounds with confidence scores:")
    for sound, score in zip(components, scores):
        print(f"  {sound}: {score:.2%}")


def example_3_speech_recognition():
    """Example: Speech to text transcription"""
    
    # Use speech recognition model
    analyzer = SoundComponentExtractor(analysis_type="speech_recognition")
    
    results = analyzer.extract_components("path/to/speech.wav")
    
    # Get transcription
    if 'transcription' in results['components']['metadata']:
        print(f"Transcription: {results['components']['metadata']['transcription']}")


def example_4_with_audio_features():
    """Example: Extract components with detailed audio features"""
    
    # Load audio file
    samples, sr = load("path/to/your/audio.wav")
    
    # Initialize analyzer
    analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
    
    # Get analysis with audio features (MFCC, spectral analysis, etc.)
    results = analyzer.extract_components_with_features(
        audio_path="path/to/your/audio.wav",
        samples=samples,
        sr=sr
    )
    
    # Access AI components
    components = results['components']['detected_sounds']
    
    # Access audio features
    features = results['audio_features']
    print(f"Spectral Centroid: {features.get('spectral_centroid', 'N/A')}")
    print(f"Spectral Rolloff: {features.get('spectral_rolloff', 'N/A')}")
    print(f"Zero Crossing Rate: {features.get('zero_crossing_rate', 'N/A')}")
    print(f"Detected Sounds: {components}")


def example_5_different_analysis_types():
    """Example: Compare different analysis types on same audio"""
    
    audio_file = "path/to/your/audio.wav"
    
    # Audio tagging (sound event detection)
    print("\n=== Audio Tagging ===")
    tagging_analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
    tagging_results = tagging_analyzer.extract_components(audio_file)
    print(f"Detected: {tagging_results['components']['detected_sounds']}")
    
    # Speech recognition
    print("\n=== Speech Recognition ===")
    speech_analyzer = SoundComponentExtractor(analysis_type="speech_recognition")
    speech_results = speech_analyzer.extract_components(audio_file)
    transcription = speech_results['components']['metadata'].get('transcription', 'N/A')
    print(f"Transcription: {transcription}")
    
    # Emotion detection
    print("\n=== Emotion Detection ===")
    emotion_analyzer = SoundComponentExtractor(analysis_type="emotion_detection")
    emotion_results = emotion_analyzer.extract_components(audio_file)
    print(f"Emotions: {emotion_results['components']['detected_sounds']}")
    
    # Music analysis
    print("\n=== Music Analysis ===")
    music_analyzer = SoundComponentExtractor(analysis_type="music_analysis")
    music_results = music_analyzer.extract_components(audio_file)
    print(f"Music Features: {music_results['components']['detected_sounds']}")


def example_6_custom_model():
    """Example: Use a specific custom model from HuggingFace"""
    
    # Use any HuggingFace model directly
    analyzer = AudioAnalyzer(model_name="facebook/wav2vec2-base-960h")
    
    results = analyzer.analyze_audio("path/to/your/audio.wav")
    print(json.dumps(results, indent=2, default=str))


def example_7_batch_processing():
    """Example: Process multiple audio files"""
    
    audio_files = [
        "audio1.wav",
        "audio2.wav",
        "audio3.wav"
    ]
    
    analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
    
    all_results = {}
    for audio_file in audio_files:
        try:
            results = analyzer.extract_components(audio_file)
            all_results[audio_file] = results['components']['detected_sounds']
            print(f"✓ {audio_file}: {all_results[audio_file]}")
        except Exception as e:
            print(f"✗ {audio_file}: Error - {e}")
    
    # Save results
    with open("analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


def example_8_integration_with_edit():
    """Example: Combine AI analysis with audio editing"""
    
    from audiopy_ai.edit import trim, normalize, apply_gain
    
    # Load audio
    samples, sr = load("path/to/your/audio.wav")
    
    # Analyze components
    analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
    results = analyzer.extract_components_with_features(
        audio_path="path/to/your/audio.wav",
        samples=samples,
        sr=sr
    )
    
    # Get detected components
    detected_sounds = results['components']['detected_sounds']
    print(f"Detected sounds: {detected_sounds}")
    
    # Apply edits based on analysis
    # For example: normalize if too quiet
    processed = normalize(samples)
    
    # Apply fade for smooth transitions
    from audiopy_ai.edit import fade
    processed = fade(processed, sr, fade_in_s=1.0, fade_out_s=1.0)
    
    # Save processed audio
    save("output_processed.wav", processed, sr)
    print(f"Analysis: {detected_sounds}")
    print("Audio saved to output_processed.wav")


if __name__ == "__main__":
    # Run examples (uncomment to test)
    
    # Basic example
    print("=" * 50)
    print("Example 1: Basic Analysis")
    print("=" * 50)
    # example_1_basic_analysis()
    
    # With API key
    print("\n" + "=" * 50)
    print("Example 2: With API Key")
    print("=" * 50)
    # example_2_with_api_key()
    
    # Speech recognition
    print("\n" + "=" * 50)
    print("Example 3: Speech Recognition")
    print("=" * 50)
    # example_3_speech_recognition()
    
    # With audio features
    print("\n" + "=" * 50)
    print("Example 4: With Audio Features")
    print("=" * 50)
    # example_4_with_audio_features()
    
    # Different analysis types
    print("\n" + "=" * 50)
    print("Example 5: Different Analysis Types")
    print("=" * 50)
    # example_5_different_analysis_types()
    
    # Custom model
    print("\n" + "=" * 50)
    print("Example 6: Custom HuggingFace Model")
    print("=" * 50)
    # example_6_custom_model()
    
    # Batch processing
    print("\n" + "=" * 50)
    print("Example 7: Batch Processing")
    print("=" * 50)
    # example_7_batch_processing()
    
    # Integration with editing
    print("\n" + "=" * 50)
    print("Example 8: Integration with Audio Editing")
    print("=" * 50)
    # example_8_integration_with_edit()
    
    print("\nUncomment the examples you want to run in the __main__ section!")
