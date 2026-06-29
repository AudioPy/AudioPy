# Step 1: Import required modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiopy_ai.io import load, save
from audiopy_ai.edit import trim
import os

# Step 2: Specify the full path to your audio file
# 👉 Change this to your own file path
input_path = r"Janam Janam_new Karaoke.mp3"

# Step 3: Check if file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ File not found at: {input_path}")
print(f"✅ Found file: {input_path}")

# Step 4: Load the audio file
samples, sr = load(input_path)
duration = len(samples) / sr
print(f"🎵 Sample rate: {sr} Hz")
print(f"📏 Duration: {duration:.2f} seconds")

# Step 5: Ask for start and end times
start_s = float(input("Enter start time (in seconds): "))
end_s = float(input("Enter end time (in seconds): "))

# Validate range
if start_s < 0 or end_s > duration or start_s >= end_s:
    raise ValueError("❌ Invalid start/end times. Please check your input range.")

# Step 6: Trim the audio
trimmed = trim(samples, sr, start_s, end_s)
print(f"✂️ Trimmed segment duration: {len(trimmed)/sr:.2f} seconds")

# Step 7: Save trimmed audio
output_path = r"D:/Documents/College PICT/Sem7/B.E. Project/Project1/AudioPy/tests/output.wav"  # You can change this path
save(output_path, trimmed, sr)
print(f"💾 Trimmed file saved at: {output_path}")
