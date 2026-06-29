import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# --- Import your AudioPy package (make sure it's installed: pip install -e .) ---
from audiopy_ai.io import load, save
from audiopy_ai.edit import trim, concat, apply_gain, normalize, fade, reverse, rms_db
from audiopy_ai.effects.filters import butter_lowpass, butter_highpass, apply_sos_filter
from audiopy_ai.effects.reverb import reverb
from audiopy_ai.effects.time import time_stretch
# NOTE: If you added a pitch_shift module later, you can import it:
# from audiopy_ai.effects.pitch import pitch_shift

# --- Demo configuration ---
INPUT_PATH = r"Mere Samnewali Khidki Mein Happy Padosan 128 Kbps.mp3"  # <-- put a path here like r"C:\Users\Adarsh\Music\input.mp3"
OUTPUT_DIR = "D:/Documents/College PICT/Sem7/B.E. Project/Project1/AudioPy/tests/output1.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_sine(freq=440, sr=22050, duration=3.0, amplitude=0.4):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t), sr


def save_and_report(name, samples, sr):
    out_path = os.path.join(OUTPUT_DIR, name)
    save(out_path, samples, sr)
    print(f"✓ Saved: {out_path}  (duration {len(samples)/sr:.2f}s, sr={sr})")
    return out_path


def plot_spectrogram(samples, sr, title="Frequency Spectrum"):
    """
    Plot the frequency spectrum of an audio signal using FFT.

    Parameters:
    - samples : np.ndarray
        Audio waveform (time-domain samples)
    - sr : int
        Sampling rate in Hz
    - title : str
        Title for the plot
    """

    # Step 1: Compute the Fast Fourier Transform (FFT)
    S = np.abs(np.fft.rfft(samples))          # Magnitude spectrum
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)  # Frequency axis (Hz)

    # Step 2: Create the plot
    plt.figure(figsize=(8, 4))
    plt.semilogy(freqs, S + 1e-9)  # log scale for amplitude (helps visibility)

    # Step 3: Label the axes
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (log scale)")  # ✅ Added Y-axis label here

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # --------- 0) Load or generate test audio ---------
    if INPUT_PATH and os.path.exists(INPUT_PATH):
        samples, sr = load(INPUT_PATH)
        print(f"Loaded input file: {INPUT_PATH} (duration {len(samples)/sr:.2f}s, sr={sr})")
    else:
        samples, sr = generate_sine(freq=440, sr=22050, duration=4.0, amplitude=0.4)
        print(f"No input file found. Generated sine wave (duration {len(samples)/sr:.2f}s, sr={sr})")
        save_and_report("demo_sine.wav", samples, sr)

    # Ensure mono 1-D arrays (AudioPy load by default returns mono if designed so)
    samples = np.asarray(samples, dtype=np.float32)

    # --------- 1) Trim ---------
    start_s, end_s = 30.0, 60.5
    t_segment = trim(samples, sr, start_s, end_s)
    print(f"[Trim] Kept {start_s}s to {end_s}s → duration {len(t_segment)/sr:.2f}s (use case: extract speech segment)")
    save_and_report("demo_trim.wav", t_segment, sr)

    # --------- 2) Concat (join) ---------
    concat_demo = concat([t_segment, t_segment])  # repeat twice
    print(f"[Concat] Joined two segments → duration {len(concat_demo)/sr:.2f}s (use case: stitch clips)")
    save_and_report("demo_concat.wav", concat_demo, sr)

    # --------- 3) Gain & Normalize ---------
    louder = apply_gain(t_segment, 6.0)  # +6 dB
    print(f"[Gain] +6 dB applied; RMS before={rms_db(t_segment):.2f} dB, after={rms_db(louder):.2f} dB")
    save_and_report("demo_gain.wav", louder, sr)

    normalized = normalize(t_segment, target_db=-18.0)
    print(f"[Normalize] Target -18 dB RMS; RMS after={rms_db(normalized):.2f} dB (use case: podcast leveling)")
    save_and_report("demo_normalize.wav", normalized, sr)

    # --------- 4) Fade In / Fade Out ---------
    faded = fade(normalized, sr, fade_in_s=0.5, fade_out_s=0.5)
    print(f"[Fade] Applied 0.5s in/out fades (use case: smooth transitions)")
    save_and_report("demo_fade.wav", faded, sr)

    # --------- 5) Reverse ---------
    reversed_audio = reverse(t_segment)
    print(f"[Reverse] Reversed audio (use case: creative sound design)")
    save_and_report("demo_reverse.wav", reversed_audio, sr)

    # --------- 6) Time-stretch (naive resampling) ---------
    faster = time_stretch(t_segment, rate=1.25)  # 25% faster
    slower = time_stretch(t_segment, rate=0.8)   # 20% slower
    print(f"[Time-stretch] Faster length={len(faster)/sr:.2f}s, Slower length={len(slower)/sr:.2f}s (use case: match clip durations)")
    save_and_report("demo_faster.wav", faster, sr)
    save_and_report("demo_slower.wav", slower, sr)

    # --------- 7) Pitch shift (crude via resampling) ---------
    # crude pitch shift: resample to change pitch (alters duration); for better results use librosa later
    def crude_pitch_shift(samples_in, sr_in, semitones=2):
        # resample ratio = 2^(semitones/12)
        ratio = 2 ** (semitones / 12.0)
        from scipy.signal import resample
        n_new = int(len(samples_in) / ratio)
        return resample(samples_in, n_new)

    pitched = crude_pitch_shift(t_segment, sr, semitones=3)
    print(f"[Pitch] Crude pitch shift +3 semitones (duration {len(pitched)/sr:.2f}s). Note: uses resampling (changes speed).")
    save_and_report("demo_pitch_crude.wav", pitched, sr)

    # --------- 8) Filters (low-pass and high-pass) ---------
    lp_sos = butter_lowpass(sr, cutoff_hz=3000, order=4)
    hp_sos = butter_highpass(sr, cutoff_hz=300, order=4)
    lp_out = apply_sos_filter(t_segment, lp_sos)
    hp_out = apply_sos_filter(t_segment, hp_sos)
    print("[Filter] Applied low-pass (3kHz) and high-pass (300Hz)")
    save_and_report("demo_lowpass.wav", lp_out, sr)
    save_and_report("demo_highpass.wav", hp_out, sr)

    # --------- 9) Reverb (convolution) ---------
    # Create a tiny synthetic impulse response (short reverb) if none provided
    ir = np.zeros(int(0.5 * sr))
    ir[0] = 1.0
    # simple exponential tail
    tail = np.exp(-np.linspace(0, 3.0, len(ir)))
    ir *= tail
    reverb_out = reverb(t_segment, ir, wet=0.35)
    print("[Reverb] Applied convolution reverb with synthetic IR (use case: add room ambience)")
    save_and_report("demo_reverb.wav", reverb_out, sr)

    # --------- 10) Spectrogram visualization (plot) ---------
    print("[Viz] Plotting spectrogram of trimmed segment (visual check)")
    plot_spectrogram(t_segment, sr, title="Trimmed Segment Spectral Magnitude")

    # Final message
    print("\nAll demos complete. Files saved in the 'outputs/' folder.")
    print("Real-world suggestions:")
    print(" - For high-quality pitch/time-stretch, consider adding librosa phase-vocoder in Stage-2.")
    print(" - Use FLOAT WAV save for near lossless testing, or PCM16 for distribution size.")

if __name__ == "__main__":
    main()