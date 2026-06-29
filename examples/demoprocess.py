# examples/demo_process.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiopy_ai.io import load, save
from audiopy_ai.edit import trim, normalize, fade, concat, apply_gain
from audiopy_ai.effects.reverb import reverb
from audiopy_ai.effects.filters import butter_lowpass, apply_sos_filter
import numpy as np

s, sr = load("inputs/lecture.wav")               # 1) load
part = trim(s, sr, 30, 80)                       # 2) trim 30s-80s
part = normalize(part, -18.0)                    # 3) normalize
part = fade(part, sr, 1, 1)                      # 4) fade
lp = butter_lowpass(sr, 8000)                    # 5) lowpass to remove hiss
part = apply_sos_filter(part, lp)
ir, _ = load("irs/room1.wav")                    # room IR
out = reverb(part, ir, wet=0.15)                 # 6) reverb
save("outputs/demo_processed.wav", out, sr)      # 7) save
print("Saved outputs/demo_processed.wav")
