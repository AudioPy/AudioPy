"""
Simple Usage Example - Copy and Paste Ready!
Run this file to see audiopy_ai in action

python simple_usage_example.py
"""

import sys
import json
from pathlib import Path
import tempfile
import numpy as np

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError with emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import audiopy_ai
from audiopy_ai.io import load, save
from audiopy_ai.edit import normalize, fade
from audiopy_ai.ai_analysis import SoundComponentExtractor


def example_1_basic_analysis():
    """Example 1: Analyze an audio file"""
    print("\n" + "="*70)
    print("Example 1: Basic Audio Analysis")
    print("="*70 + "\n")
    
    # Create a test audio file
    print("Creating test audio...")
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "test.wav"
        save(str(audio_path), samples, sr)
        print(f"✓ Saved test audio\n")
        
        # Initialize analyzer
        print("Initializing AI Analyzer...")
        analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
        print("✓ Analyzer ready\n")
        
        # Analyze
        print("Analyzing audio (this may take a moment on first run)...")
        results = analyzer.extract_components(str(audio_path))
        print("✓ Analysis complete!\n")
        
        # Display results
        if 'components' in results:
            sounds = results['components']['detected_sounds']
            scores = results['components']['confidence_scores']
            
            print("🔊 Detected Sounds:")
            for sound, score in zip(sounds[:5], scores[:5]):
                print(f"   • {sound}: {score*100:.1f}%")
        
        print()


def example_2_mp4_analysis():
    """Example 2: Analyze MP4 file (requires real MP4)"""
    print("\n" + "="*70)
    print("Example 2: Analyze MP4 File")
    print("="*70 + "\n")
    
    print("To analyze an MP4 file, use:")
    print("   python test_mp4_analysis.py your_video.mp4\n")
    print("Or use the interactive tool:")
    print("   python interactive_test.py\n")


def example_3_with_features():
    """Example 3: Analysis with audio features"""
    print("\n" + "="*70)
    print("Example 3: Analysis with Audio Features")
    print("="*70 + "\n")
    
    # Create test audio
    print("Creating test audio...")
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "test.wav"
        save(str(audio_path), samples, sr)
        print(f"✓ Saved test audio\n")
        
        # Analyze with features
        print("Analyzing with audio features...")
        analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
        results = analyzer.extract_components_with_features(
            audio_path=str(audio_path),
            samples=samples,
            sr=sr
        )
        print("✓ Analysis complete!\n")
        
        # Display features
        if 'audio_features' in results and results['audio_features']:
            features = results['audio_features']
            print("📈 Audio Features:")
            
            if 'spectral_centroid' in features:
                print(f"   • Spectral Centroid: {features['spectral_centroid']:.2f} Hz")
            if 'spectral_rolloff' in features:
                print(f"   • Spectral Rolloff: {features['spectral_rolloff']:.2f} Hz")
            if 'zero_crossing_rate' in features:
                print(f"   • Zero Crossing Rate: {features['zero_crossing_rate']:.4f}")
            if 'onset_strength' in features:
                print(f"   • Onset Strength: {features['onset_strength']:.4f}")
        
        print()


def example_4_different_analysis_types():
    """Example 4: Try different analysis types"""
    print("\n" + "="*70)
    print("Example 4: Different Analysis Types")
    print("="*70 + "\n")
    
    analysis_types = [
        ("audio_tagging", "Detect sound events"),
        ("speech_recognition", "Transcribe speech"),
        ("emotion_detection", "Detect emotions"),
        ("music_analysis", "Analyze music"),
    ]
    
    print("Available analysis types:\n")
    for atype, description in analysis_types:
        print(f"   • {atype:<25} - {description}")
    
    print("\nExample usage:")
    print("   analyzer = SoundComponentExtractor(analysis_type='speech_recognition')")
    print("   results = analyzer.extract_components('audio.wav')\n")


def example_5_editing_and_analysis():
    """Example 5: Edit audio and analyze"""
    print("\n" + "="*70)
    print("Example 5: Edit and Analyze")
    print("="*70 + "\n")
    
    # Create test audio
    print("Creating test audio...")
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "test.wav"
        save(str(audio_path), samples, sr)
        
        # Load
        print("✓ Created audio")
        loaded, sr = load(str(audio_path))
        
        # Process
        print("Processing audio...")
        processed = normalize(loaded)
        processed = fade(processed, sr, fade_in_s=0.5, fade_out_s=0.5)
        print("✓ Applied normalize + fade\n")
        
        # Save processed
        output_path = Path(tmpdir) / "processed.wav"
        save(str(output_path), processed, sr)
        print("✓ Saved processed audio\n")
        
        # Analyze
        print("Analyzing processed audio...")
        analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
        try:
            results = analyzer.extract_components(str(output_path))
            print("✓ Analysis complete\n")
        except Exception as e:
            print(f"Note: {e}\n")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("audiopy_ai Simple Usage Examples")
    print("="*70)
    
    try:
        # Example 1
        example_1_basic_analysis()
        
        # Example 2
        example_2_mp4_analysis()
        
        # Example 3
        example_3_with_features()
        
        # Example 4
        example_4_different_analysis_types()
        
        # Example 5
        example_5_editing_and_analysis()
        
        print("\n" + "="*70)
        print("✅ Examples completed!")
        print("="*70)
        print("\nYou can now:")
        print("   1. Test with your own MP4 file:")
        print("      python test_mp4_analysis.py your_video.mp4")
        print("\n   2. Use the interactive tool:")
        print("      python interactive_test.py")
        print("\n   3. Integrate audiopy_ai into your project:")
        print("      from audiopy_ai.ai_analysis import SoundComponentExtractor")
        print("      analyzer = SoundComponentExtractor()")
        print("      results = analyzer.extract_components('audio.wav')")
        print("\n📖 Documentation: Check AI_ANALYSIS_GUIDE.md for detailed API\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure dependencies are installed:")
        print("   pip install transformers torch librosa numpy scipy soundfile\n")
        raise


if __name__ == "__main__":
    main()
