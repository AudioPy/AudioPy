"""
Test audiopy_ai AI Model with MP4 Files
Upload an MP4 file and see the AI-powered sound component analysis

Usage:
    python test_mp4_analysis.py your_video.mp4
    
Or without arguments to use a test file:
    python test_mp4_analysis.py
"""

import sys
import os
from pathlib import Path
import json
import tempfile
import numpy as np
from typing import Tuple

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError with emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from audiopy_ai.io import load, save
from audiopy_ai.ai_analysis import SoundComponentExtractor


def extract_audio_from_mp4(mp4_path: str, output_wav_path: str) -> bool:
    """
    Extract audio from MP4 file using ffmpeg
    
    Args:
        mp4_path: Path to MP4 file
        output_wav_path: Path to save extracted audio
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', mp4_path,
            '-q:a', '9',  # Audio quality
            '-n',  # Don't overwrite
            output_wav_path
        ]
        
        # Run ffmpeg silently
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ FFmpeg not found. Install with:")
        print("   Windows (Chocolatey): choco install ffmpeg")
        print("   Windows (Winget): winget install ffmpeg")
        print("   Mac: brew install ffmpeg")
        print("   Linux: apt-get install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Audio extraction timed out")
        return False
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False


def analyze_mp4_audio(mp4_path: str, analysis_type: str = "audio_tagging") -> dict:
    """
    Analyze audio from MP4 file using AI models
    
    Args:
        mp4_path: Path to MP4 file
        analysis_type: Type of analysis (audio_tagging, speech_recognition, etc.)
    
    Returns:
        Dictionary with analysis results
    """
    mp4_path = Path(mp4_path)
    
    # Verify file exists
    if not mp4_path.exists():
        print(f"❌ File not found: {mp4_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"audiopy_ai AI Analysis - MP4 Audio Test")
    print(f"{'='*70}\n")
    
    print(f"📁 Input File: {mp4_path.name}")
    print(f"📊 File Size: {mp4_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"🔍 Analysis Type: {analysis_type}\n")
    
    # Create temporary directory for extracted audio
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "extracted_audio.wav"
        
        # Step 1: Extract audio from MP4
        print("Step 1️⃣  Extracting audio from MP4...")
        if not extract_audio_from_mp4(str(mp4_path), str(wav_path)):
            print("❌ Failed to extract audio")
            return None
        
        if not wav_path.exists():
            print("❌ Audio extraction failed - file not created")
            return None
        
        wav_size = wav_path.stat().st_size / (1024*1024)
        print(f"   ✅ Audio extracted: {wav_size:.2f} MB\n")
        
        # Step 2: Load and analyze audio
        print("Step 2️⃣  Loading audio file...")
        try:
            samples, sr = load(str(wav_path))
            duration = len(samples) / sr
            print(f"   ✅ Loaded: {len(samples):,} samples at {sr}Hz")
            print(f"   📈 Duration: {duration:.2f} seconds\n")
        except Exception as e:
            print(f"   ❌ Failed to load audio: {e}")
            return None
        
        # Step 3: Initialize analyzer
        print("Step 3️⃣  Initializing AI Analyzer...")
        try:
            analyzer = SoundComponentExtractor(analysis_type=analysis_type)
            print(f"   ✅ Analyzer ready (Model: {analyzer.analysis_type})\n")
        except Exception as e:
            print(f"   ❌ Failed to initialize analyzer: {e}")
            return None
        
        # Step 4: Run analysis
        print("Step 4️⃣  Analyzing audio...")
        print("   ⏳ Processing (this may take a moment on first run)...\n")
        
        try:
            results = analyzer.extract_components_with_features(
                audio_path=str(wav_path),
                samples=samples,
                sr=sr
            )
            print("   ✅ Analysis complete!\n")
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            return None
        
        return results


def display_results(results: dict):
    """Display analysis results in a formatted way"""
    
    print(f"{'='*70}")
    print("📊 ANALYSIS RESULTS")
    print(f"{'='*70}\n")
    
    # 1. Detected Sound Components
    if 'components' in results:
        components = results['components']
        
        if 'detected_sounds' in components and components['detected_sounds']:
            print("🔊 DETECTED SOUND COMPONENTS:")
            print("-" * 70)
            
            sounds = components['detected_sounds']
            scores = components['confidence_scores']
            
            for i, (sound, score) in enumerate(zip(sounds, scores), 1):
                percentage = score * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {i}. {sound:<30} [{bar}] {percentage:.1f}%")
            
            print()
    
    # 2. Audio Features
    if 'audio_features' in results and results['audio_features']:
        print("📈 AUDIO FEATURES:")
        print("-" * 70)
        
        features = results['audio_features']
        
        # Spectral features
        if 'spectral_centroid' in features:
            print(f"  Spectral Centroid (brightness): {features['spectral_centroid']:.2f} Hz")
        
        if 'spectral_rolloff' in features:
            print(f"  Spectral Rolloff:               {features['spectral_rolloff']:.2f} Hz")
        
        if 'zero_crossing_rate' in features:
            print(f"  Zero Crossing Rate (noisiness): {features['zero_crossing_rate']:.4f}")
        
        if 'onset_strength' in features:
            print(f"  Onset Strength (rhythm):        {features['onset_strength']:.4f}")
        
        # MFCC
        if 'mfcc_mean' in features:
            mfcc_mean = features['mfcc_mean']
            print(f"  MFCC Mean (13 coefficients):    {[f'{x:.3f}' for x in mfcc_mean[:5]]}...")
        
        # Chroma
        if 'chroma_mean' in features:
            print(f"  Chroma Features (pitch):        Available")
        
        print()
    
    # 3. Metadata
    if 'components' in results:
        metadata = results['components'].get('metadata', {})
        if metadata:
            print("📋 METADATA:")
            print("-" * 70)
            for key, value in metadata.items():
                if isinstance(value, str):
                    display_value = value[:70] + "..." if len(value) > 70 else value
                    print(f"  {key}: {display_value}")
            print()
    
    # 4. Model Information
    print("🤖 MODEL INFORMATION:")
    print("-" * 70)
    print(f"  Model:     {results.get('model', 'Unknown')}")
    print(f"  Task:      {results.get('task', 'Unknown')}")
    print()
    
    # 5. Raw Results (JSON)
    print("📄 RAW RESULTS (JSON):")
    print("-" * 70)
    print(json.dumps(results, indent=2, default=str)[:2000])
    print("\n... (truncated)\n")


def create_test_mp4():
    """Create a simple test MP4 file with audio"""
    try:
        from moviepy.editor import AudioFileClip, ImageClip
        import numpy as np
    except ImportError:
        print("⚠️  moviepy not installed. Using audio-only approach.\n")
        return None
    
    # This would require moviepy, which adds complexity
    # For now, we'll just use audio files
    return None


def main():
    """Main test function"""
    
    # Get MP4 file path from command line
    if len(sys.argv) > 1:
        mp4_file = sys.argv[1]
    else:
        print("\n" + "="*70)
        print("🎬 audiopy_ai MP4 Audio Analysis Test")
        print("="*70 + "\n")
        print("Usage: python test_mp4_analysis.py <mp4_file>\n")
        print("Example: python test_mp4_analysis.py video.mp4\n")
        
        print("No MP4 file provided. Creating test audio instead...\n")
        
        # Create a test WAV file to demonstrate
        with tempfile.TemporaryDirectory() as tmpdir:
            test_wav = Path(tmpdir) / "test_audio.wav"
            
            # Generate test audio
            sr = 16000
            duration = 3
            t = np.linspace(0, duration, sr * duration)
            
            # Mix of frequencies to simulate different sounds
            samples = (
                0.2 * np.sin(2 * np.pi * 440 * t) +     # A4 note
                0.2 * np.sin(2 * np.pi * 880 * t) +     # A5 note
                0.1 * np.sin(2 * np.pi * 1760 * t)      # A6 note
            ).astype(np.float32)
            
            save(str(test_wav), samples, sr)
            
            # Analyze
            try:
                analyzer = SoundComponentExtractor(analysis_type="audio_tagging")
                results = analyzer.extract_components_with_features(
                    audio_path=str(test_wav),
                    samples=samples,
                    sr=sr
                )
                
                display_results(results)
                
            except Exception as e:
                print(f"Analysis error: {e}\n")
                print("Make sure transformers and torch are installed:")
                print("  pip install transformers torch librosa\n")
        
        return
    
    # Analyze provided MP4 file
    results = analyze_mp4_audio(mp4_file, analysis_type="audio_tagging")
    
    if results:
        display_results(results)
        
        # Save results to file
        output_file = Path(mp4_file).stem + "_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {output_file}\n")
    else:
        print("❌ Analysis failed\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis cancelled by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
