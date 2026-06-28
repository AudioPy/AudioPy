import audiopy as ap
import numpy as np

print("🎵 Starting audiopy demo...")

# 1. Generate some synthetic audio to simulate a real audio file
print("\n[1/5] Creating a synthetic sound (440Hz Sine Wave)...")
sample_rate = 22050
duration = 2.0 # seconds
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Save the synthetic audio to a file
ap.save("input_audio.wav", audio_data, sample_rate)
print("      Saved to 'input_audio.wav'")

# 2. Use audiopy to load the audio file
print("\n[2/5] Loading 'input_audio.wav' using audiopy...")
y, sr = ap.load("input_audio.wav")
print(f"      Loaded {ap.duration(y, sr):.2f} seconds of audio at {sr} Hz")

# 3. Apply an effect: Let's add some reverb and an echo
print("\n[3/5] Applying Reverb and Echo effects...")
y_reverb = ap.reverb(y, sr, room_scale=0.8, wet_level=0.5)
y_echo = ap.echo(y_reverb, sr, delay=0.3, decay=0.4)

# 4. Extract some features
print("\n[4/5] Extracting audio features...")
tempo = ap.tempo(y_echo, sr)
print(f"      Estimated tempo: {tempo:.1f} BPM")
rms = ap.rms_energy(y_echo)
print(f"      Mean RMS Energy: {rms.mean():.4f}")

# 5. Save the final processed audio
print("\n[5/5] Saving processed audio...")
ap.save("output_processed.wav", y_echo, sr)
print("      Saved to 'output_processed.wav'")

print("\n✅ Demo completed successfully! audiopy is ready to use.")
