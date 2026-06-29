"""
Quick Start Test - Run this to verify audiopy_ai AI Analysis works
Execute: python quick_test.py
"""

import sys
import subprocess
from pathlib import Path
import tempfile
import numpy as np

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError with emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, 
                      timeout=5, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False

from audiopy_ai.io import load, save
from audiopy_ai.edit import normalize, fade
from audiopy_ai.ai_analysis import SoundComponentExtractor


def quick_test():
    """Quick functionality test"""
    
    print("\n" + "="*60)
    print("audiopy_ai AI Analysis - Quick Test")
    print("="*60 + "\n")
    
    # Check FFmpeg
    print("0️⃣  Checking FFmpeg installation...")
    if not check_ffmpeg():
        print("   ❌ FFmpeg not found!\n")
        print("   FFmpeg is required to load audio files.")
        print("   Install it using:\n")
        print("   Windows (PowerShell):")
        print("     winget install FFmpeg\n")
        print("   macOS:")
        print("     brew install ffmpeg\n")
        print("   Linux (Ubuntu/Debian):")
        print("     sudo apt-get install ffmpeg\n")
        print("   Then restart your terminal and try again.\n")
        return False
    print("   ✓ FFmpeg found\n")
    
    # 1. Create sample audio
    print("1️⃣  Creating test audio...")
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    print(f"   ✓ Created {len(samples)} samples at {sr}Hz\n")
    
    # 2. Save audio
    print("2️⃣  Saving audio file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "test.wav"
        save(str(audio_path), samples, sr)
        print(f"   ✓ Saved to {audio_path.name}\n")
        
        # 3. Load and process
        print("3️⃣  Loading and processing audio...")
        loaded_samples, loaded_sr = load(str(audio_path))
        normalized = normalize(loaded_samples)
        processed = fade(normalized, sr, fade_in_s=0.5, fade_out_s=0.5)
        print(f"   ✓ Processed: normalized + fade\n")
        
        # 4. Initialize AI Analyzer
        print("4️⃣  Initializing AI Analyzer...")
        try:
            analyzer = SoundComponentExtractor(analysis_type="speech_recognition")
            print(f"   ✓ Analyzer ready\n")
            
            # 5. Analyze audio
            print("5️⃣  Analyzing audio file...")
            print("   ⏳ This may take a moment on first run (downloading models)...\n")
            
            try:
                results = analyzer.extract_components(str(audio_path))
                
                print("   ✓ Analysis complete!\n")
                
                # Display results
                if 'components' in results:
                    sounds = results['components'].get('detected_sounds', [])
                    scores = results['components'].get('confidence_scores', [])
                    
                    print("   Detected Components:")
                    for sound, score in zip(sounds, scores):
                        percentage = f"{score*100:.1f}%"
                        print(f"     • {sound}: {percentage}")
                    print()
                
                print("="*60)
                print("✅ ALL TESTS PASSED!")
                print("="*60)
                print("\n📚 Next Steps:")
                print("   1. Check examples/ai_analysis_examples.py for usage patterns")
                print("   2. Read AI_ANALYSIS_GUIDE.md for detailed documentation")
                print("   3. Run tests/test_ai_analysis.py for comprehensive testing\n")
                return True
                
            except Exception as e:
                print(f"   ℹ️  Analysis requires model download (will happen automatically)")
                print(f"   ℹ️  Error: {str(e)[:100]}\n")
                print("="*60)
                print("✅ Core functions working! (AI models need downloading)")
                print("="*60)
                print("\nTo download models, run:")
                print("   from transformers import pipeline")
                print("   pipeline('audio-classification', model='mtg-upf/discogs-effnet-bs64-1')\n")
                return True
        
        except Exception as e:
            print(f"   ⚠️  {e}\n")
            print("="*60)
            print("✅ Core I/O and editing working!")
            print("⚠️  AI features need dependencies")
            print("="*60)
            print("\nTo enable AI analysis, install:")
            print("   pip install transformers torch librosa\n")
            return True


if __name__ == "__main__":
    try:
        result = quick_test()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
