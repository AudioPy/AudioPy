"""
Interactive audiopy_ai AI Analysis Tool
A user-friendly CLI for testing audio/MP4 files with AI models

Usage: python interactive_test.py
"""

import sys
from pathlib import Path
import json
import tempfile
import subprocess
from typing import Optional

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError with emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from audiopy_ai.io import load, save
from audiopy_ai.ai_analysis import SoundComponentExtractor


class Colors:
    """Terminal colors"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'


def print_header(title):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_menu():
    """Display main menu"""
    print_header("audiopy_ai AI Analysis - Interactive Tool")
    
    print(f"{Colors.CYAN}Select an option:{Colors.RESET}\n")
    print("  1️⃣  Analyze MP4/Video File")
    print("  2️⃣  Analyze Audio File (WAV, MP3, etc.)")
    print("  3️⃣  Test with Synthetic Audio")
    print("  4️⃣  Change Analysis Type")
    print("  5️⃣  View Previous Results")
    print("  6️⃣  Run Diagnostic Tests")
    print("  0️⃣  Exit")
    print()


def get_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default"""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    value = input(f"{Colors.CYAN}{prompt}{Colors.RESET}")
    return value.strip() or default


def extract_audio_from_mp4(mp4_path: str, output_wav_path: str) -> bool:
    """Extract audio from MP4 using ffmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-i', mp4_path,
            '-q:a', '9',
            '-n',
            output_wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"{Colors.RED}❌ FFmpeg not found. Install with:{Colors.RESET}")
        print(f"   Windows: winget install ffmpeg")
        print(f"   Mac: brew install ffmpeg")
        print(f"   Linux: sudo apt-get install ffmpeg")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        return False


def analyze_audio_file(
    audio_path: str,
    analysis_type: str = "audio_tagging",
    extract_features: bool = True
) -> Optional[dict]:
    """Analyze an audio file"""
    
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        print(f"{Colors.RED}❌ File not found: {audio_path}{Colors.RESET}\n")
        return None
    
    print(f"\n{Colors.YELLOW}⏳ Processing...{Colors.RESET}\n")
    
    try:
        # Load audio
        print(f"{Colors.CYAN}Loading audio...{Colors.RESET}", end=" ", flush=True)
        samples, sr = load(str(audio_path))
        duration = len(samples) / sr
        print(f"{Colors.GREEN}✓{Colors.RESET}")
        
        # Initialize analyzer
        print(f"{Colors.CYAN}Initializing analyzer...{Colors.RESET}", end=" ", flush=True)
        analyzer = SoundComponentExtractor(analysis_type=analysis_type)
        print(f"{Colors.GREEN}✓{Colors.RESET}")
        
        # Analyze
        print(f"{Colors.CYAN}Analyzing audio (this may take a moment)...{Colors.RESET}")
        
        if extract_features:
            results = analyzer.extract_components_with_features(
                audio_path=str(audio_path),
                samples=samples,
                sr=sr
            )
        else:
            results = analyzer.extract_components(str(audio_path))
        
        print(f"{Colors.GREEN}✓ Analysis complete!{Colors.RESET}")
        
        # Display results
        display_results(results, audio_path, duration)
        
        # Save results
        output_file = audio_path.stem + "_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{Colors.GREEN}✅ Results saved to: {output_file}{Colors.RESET}\n")
        
        return results
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}\n")
        return None


def display_results(results: dict, audio_path: Path, duration: float):
    """Display analysis results"""
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}📊 ANALYSIS RESULTS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.RESET}\n")
    
    # File info
    print(f"{Colors.CYAN}📁 File:{Colors.RESET} {audio_path.name}")
    print(f"{Colors.CYAN}⏱️  Duration:{Colors.RESET} {duration:.2f} seconds")
    print(f"{Colors.CYAN}🎵 Sample Rate:{Colors.RESET} 16000 Hz\n")
    
    # Detected sounds
    if 'components' in results:
        components = results['components']
        sounds = components.get('detected_sounds', [])
        scores = components.get('confidence_scores', [])
        
        if sounds:
            print(f"{Colors.CYAN}🔊 DETECTED SOUND COMPONENTS:{Colors.RESET}")
            print("-" * 70)
            
            for i, (sound, score) in enumerate(zip(sounds, scores), 1):
                percentage = score * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {i}. {sound:<25} [{bar}] {percentage:5.1f}%")
            
            print()
    
    # Audio features
    if 'audio_features' in results and results['audio_features']:
        features = results['audio_features']
        print(f"{Colors.CYAN}📈 AUDIO FEATURES:{Colors.RESET}")
        print("-" * 70)
        
        if 'spectral_centroid' in features:
            print(f"  Spectral Centroid:  {features['spectral_centroid']:>10.2f} Hz")
        if 'spectral_rolloff' in features:
            print(f"  Spectral Rolloff:   {features['spectral_rolloff']:>10.2f} Hz")
        if 'zero_crossing_rate' in features:
            print(f"  Zero Crossing Rate: {features['zero_crossing_rate']:>10.4f}")
        if 'onset_strength' in features:
            print(f"  Onset Strength:     {features['onset_strength']:>10.4f}")
        
        print()


def select_analysis_type() -> str:
    """Let user select analysis type"""
    print_header("Select Analysis Type")
    
    types = {
        '1': 'audio_tagging',
        '2': 'speech_recognition',
        '3': 'emotion_detection',
        '4': 'music_analysis'
    }
    
    print(f"{Colors.CYAN}Analysis Types:{Colors.RESET}\n")
    print("  1️⃣  Audio Tagging (detect sounds)")
    print("  2️⃣  Speech Recognition (transcribe)")
    print("  3️⃣  Emotion Detection")
    print("  4️⃣  Music Analysis\n")
    
    choice = get_input("Select type", "1")
    return types.get(choice, 'audio_tagging')


def run_diagnostics():
    """Run diagnostic tests"""
    print_header("Running Diagnostics")
    
    # 1. FFmpeg
    print(f"{Colors.CYAN}Checking FFmpeg...{Colors.RESET}", end=" ", flush=True)
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"{Colors.GREEN}✓{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗{Colors.RESET}")
    except FileNotFoundError:
        print(f"{Colors.RED}✗ Not installed{Colors.RESET}")
    
    # 2. Transformers
    print(f"{Colors.CYAN}Checking transformers...{Colors.RESET}", end=" ", flush=True)
    try:
        import transformers
        print(f"{Colors.GREEN}✓{Colors.RESET}")
    except ImportError:
        print(f"{Colors.RED}✗ Not installed{Colors.RESET}")
    
    # 3. Torch
    print(f"{Colors.CYAN}Checking torch...{Colors.RESET}", end=" ", flush=True)
    try:
        import torch
        print(f"{Colors.GREEN}✓{Colors.RESET}")
    except ImportError:
        print(f"{Colors.RED}✗ Not installed{Colors.RESET}")
    
    # 4. Librosa
    print(f"{Colors.CYAN}Checking librosa...{Colors.RESET}", end=" ", flush=True)
    try:
        import librosa
        print(f"{Colors.GREEN}✓{Colors.RESET}")
    except ImportError:
        print(f"{Colors.RED}✗ Optional (for audio features){Colors.RESET}")
    
    print()


def main():
    """Main interactive loop"""
    analysis_type = "audio_tagging"
    
    while True:
        print_menu()
        choice = get_input("Enter choice", "0")
        
        if choice == "0":
            print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.RESET}\n")
            break
        
        elif choice == "1":
            # Analyze MP4
            print_header("Analyze MP4/Video File")
            file_path = get_input("Enter MP4 file path")
            
            if not file_path:
                print(f"{Colors.RED}❌ No file provided{Colors.RESET}\n")
                continue
            
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"{Colors.RED}❌ File not found{Colors.RESET}\n")
                continue
            
            # Extract audio
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = Path(tmpdir) / "extracted.wav"
                
                print(f"\n{Colors.YELLOW}⏳ Extracting audio from MP4...{Colors.RESET}\n")
                if extract_audio_from_mp4(str(file_path), str(wav_path)):
                    print(f"{Colors.GREEN}✓ Audio extracted{Colors.RESET}")
                    
                    # Analyze
                    analyze_audio_file(str(wav_path), analysis_type, extract_features=True)
                else:
                    print(f"{Colors.RED}❌ Failed to extract audio{Colors.RESET}\n")
        
        elif choice == "2":
            # Analyze audio file
            print_header("Analyze Audio File")
            file_path = get_input("Enter audio file path")
            
            if file_path:
                analyze_audio_file(file_path, analysis_type, extract_features=True)
        
        elif choice == "3":
            # Test with synthetic audio
            print_header("Test with Synthetic Audio")
            print(f"{Colors.YELLOW}⏳ Creating synthetic audio...{Colors.RESET}\n")
            
            import numpy as np
            
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_file = Path(tmpdir) / "synthetic.wav"
                
                sr = 16000
                duration = 3
                t = np.linspace(0, duration, sr * duration)
                
                # Create mixed tones
                samples = (
                    0.2 * np.sin(2 * np.pi * 440 * t) +
                    0.2 * np.sin(2 * np.pi * 880 * t) +
                    0.1 * np.sin(2 * np.pi * 1760 * t)
                ).astype(np.float32)
                
                save(str(audio_file), samples, sr)
                
                print(f"{Colors.GREEN}✓ Synthetic audio created{Colors.RESET}")
                analyze_audio_file(str(audio_file), analysis_type, extract_features=True)
        
        elif choice == "4":
            # Change analysis type
            analysis_type = select_analysis_type()
            print(f"\n{Colors.GREEN}✓ Analysis type changed to: {analysis_type}{Colors.RESET}\n")
            input("Press Enter to continue...")
        
        elif choice == "5":
            # View previous results
            print_header("View Previous Results")
            
            from glob import glob
            results_files = glob("*_analysis.json")
            
            if not results_files:
                print(f"{Colors.YELLOW}No previous results found{Colors.RESET}\n")
                continue
            
            print(f"{Colors.CYAN}Found results files:{Colors.RESET}\n")
            for i, file in enumerate(results_files, 1):
                print(f"  {i}. {file}")
            
            choice = get_input("\nSelect file number", "1")
            try:
                selected = results_files[int(choice) - 1]
                with open(selected, 'r') as f:
                    results = json.load(f)
                
                audio_file = Path(selected).stem.replace("_analysis", "")
                display_results(results, Path(audio_file), 0)
            except (ValueError, IndexError, FileNotFoundError):
                print(f"{Colors.RED}❌ Invalid selection{Colors.RESET}\n")
        
        elif choice == "6":
            # Run diagnostics
            run_diagnostics()
            input("Press Enter to continue...")
        
        else:
            print(f"{Colors.RED}❌ Invalid choice{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Cancelled by user{Colors.RESET}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
