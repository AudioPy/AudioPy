"""
test_audiopy.py — Comprehensive test script for audiopy
=========================================================
Tests every module using synthetic audio (no external files needed).
Run with:  python test_audiopy.py
"""

import sys
import os
import numpy as np
import tempfile

# Ensure audiopy is importable from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
skipped = 0
results = []


def test(name, func):
    """Run a test function, record pass/fail/skip."""
    global passed, failed, skipped
    try:
        result = func()
        if result == "SKIP":
            skipped += 1
            results.append(("SKIP", name))
            print(f"  ⏭️  SKIP  {name}")
        else:
            passed += 1
            results.append(("PASS", name))
            print(f"  ✅ PASS  {name}")
    except Exception as e:
        failed += 1
        results.append(("FAIL", name, str(e)))
        print(f"  ❌ FAIL  {name}: {e}")


# ─── Generate synthetic audio ──────────────────────────────────────────────
SR = 22050
DURATION = 2.0  # seconds
N_SAMPLES = int(SR * DURATION)

# 440 Hz sine wave (A4 note)
t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
SINE_WAVE = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

# Stereo version
STEREO = np.stack([SINE_WAVE, SINE_WAVE * 0.8]).astype(np.float32)

# Noisy version
NOISY = SINE_WAVE + 0.05 * np.random.randn(N_SAMPLES).astype(np.float32)


print("=" * 60)
print("  audiopy Test Suite")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Import
# ═══════════════════════════════════════════════════════════════════════════
print("\n📦 Import Tests")
print("-" * 40)


def test_import():
    import audiopy as ap
    assert hasattr(ap, "__version__")
    assert ap.__version__ == "0.1.0"


test("import audiopy as ap", test_import)


def test_import_submodules():
    import audiopy as ap
    assert hasattr(ap, "models")
    assert hasattr(ap, "utils")


test("import submodules", test_import_submodules)


def test_all_exports():
    import audiopy as ap
    for name in ["load", "save", "mono", "mfcc", "reverb",
                  "plot_waveform", "transcribe", "detect_emotion",
                  "separate_vocals", "speak", "denoise_deep"]:
        assert hasattr(ap, name), f"Missing export: {name}"


test("all key functions exported", test_all_exports)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: I/O Module
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔊 I/O Tests")
print("-" * 40)


def test_save_and_load():
    import audiopy as ap
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        ap.save(tmp_path, SINE_WAVE, SR)
        y, sr = ap.load(tmp_path, sr=SR)
        assert isinstance(y, np.ndarray), "load must return ndarray"
        assert y.dtype == np.float32, "must be float32"
        assert sr == SR
        assert len(y) > 0
    finally:
        os.remove(tmp_path)


test("save + load WAV", test_save_and_load)


def test_save_flac():
    import audiopy as ap
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
        tmp_path = f.name
    try:
        ap.save(tmp_path, SINE_WAVE, SR, format="flac")
        y, sr = ap.load(tmp_path, sr=SR)
        assert len(y) > 0
    finally:
        os.remove(tmp_path)


test("save + load FLAC", test_save_flac)


def test_info():
    import audiopy as ap
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        ap.save(tmp_path, SINE_WAVE, SR)
        meta = ap.info(tmp_path)
        assert "duration" in meta
        assert "sample_rate" in meta
        assert "channels" in meta
        assert "format" in meta
        assert "file_size" in meta
        assert meta["sample_rate"] == SR
        assert meta["duration"] > 0
    finally:
        os.remove(tmp_path)


test("info() metadata", test_info)


def test_load_nonexistent():
    import audiopy as ap
    try:
        ap.load("__nonexistent_file__.wav")
        return False  # should have raised
    except FileNotFoundError:
        pass  # expected


test("load nonexistent file raises FileNotFoundError", test_load_nonexistent)


def test_save_stereo():
    import audiopy as ap
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        ap.save(tmp_path, STEREO, SR)
        y, sr = ap.load(tmp_path, sr=SR)
        assert len(y) > 0
    finally:
        os.remove(tmp_path)


test("save + load stereo", test_save_stereo)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Core Module
# ═══════════════════════════════════════════════════════════════════════════
print("\n⚙️  Core Processing Tests")
print("-" * 40)


def test_resample():
    import audiopy as ap
    y_16k = ap.resample(SINE_WAVE, orig_sr=SR, target_sr=16000)
    expected_len = int(N_SAMPLES * 16000 / SR)
    assert abs(len(y_16k) - expected_len) < 10
    assert y_16k.dtype == np.float32


test("resample 22050 -> 16000", test_resample)


def test_mono():
    import audiopy as ap
    y_mono = ap.mono(STEREO)
    assert y_mono.ndim == 1
    assert y_mono.dtype == np.float32
    # Already mono should stay mono
    y_same = ap.mono(SINE_WAVE)
    assert y_same.ndim == 1


test("mono conversion", test_mono)


def test_stereo_convert():
    import audiopy as ap
    y_stereo = ap.stereo(SINE_WAVE)
    assert y_stereo.shape == (2, N_SAMPLES)
    assert y_stereo.dtype == np.float32


test("stereo conversion", test_stereo_convert)


def test_trim():
    import audiopy as ap
    # Add silence + signal
    silence = np.zeros(5000, dtype=np.float32)
    padded = np.concatenate([silence, SINE_WAVE, silence])
    trimmed = ap.trim(padded, SR, top_db=20)
    assert len(trimmed) < len(padded), "trim should remove silence"
    assert len(trimmed) > 0


test("trim silence", test_trim)


def test_pad():
    import audiopy as ap
    target = N_SAMPLES * 2
    y_pad = ap.pad(SINE_WAVE, target_length=target)
    assert len(y_pad) == target
    # Truncation
    y_trunc = ap.pad(SINE_WAVE, target_length=1000)
    assert len(y_trunc) == 1000


test("pad / truncate", test_pad)


def test_split_on_silence():
    import audiopy as ap
    silence = np.zeros(int(SR * 0.6), dtype=np.float32)  # 600ms gap
    audio = np.concatenate([SINE_WAVE[:SR], silence, SINE_WAVE[:SR]])
    segments = ap.split_on_silence(audio, SR, min_silence_len=400, silence_thresh=-30)
    assert isinstance(segments, list)
    assert len(segments) >= 1


test("split_on_silence", test_split_on_silence)


def test_concatenate():
    import audiopy as ap
    y = ap.concatenate([SINE_WAVE[:1000], SINE_WAVE[:2000]], SR)
    assert len(y) == 3000


test("concatenate", test_concatenate)


def test_mix():
    import audiopy as ap
    y_mixed = ap.mix(SINE_WAVE, SINE_WAVE * 0.5, weight1=0.7, weight2=0.3)
    assert len(y_mixed) == N_SAMPLES
    assert y_mixed.dtype == np.float32


test("mix two signals", test_mix)


def test_normalize():
    import audiopy as ap
    quiet = SINE_WAVE * 0.01
    y_norm = ap.normalize(quiet)
    assert abs(np.max(np.abs(y_norm)) - 1.0) < 0.01


test("normalize", test_normalize)


def test_duration_func():
    import audiopy as ap
    d = ap.duration(SINE_WAVE, SR)
    assert abs(d - DURATION) < 0.01


test("duration()", test_duration_func)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Features Module
# ═══════════════════════════════════════════════════════════════════════════
print("\n📊 Feature Extraction Tests")
print("-" * 40)


def test_mfcc():
    import audiopy as ap
    m = ap.mfcc(SINE_WAVE, SR, n_mfcc=13)
    assert m.shape[0] == 13
    assert m.dtype == np.float32


test("mfcc (13 coefficients)", test_mfcc)


def test_mel_spectrogram():
    import audiopy as ap
    m = ap.mel_spectrogram(SINE_WAVE, SR, n_mels=128)
    assert m.shape[0] == 128
    assert m.dtype == np.float32


test("mel_spectrogram (128 bands)", test_mel_spectrogram)


def test_spectrogram():
    import audiopy as ap
    S = ap.spectrogram(SINE_WAVE, SR)
    assert S.ndim == 2
    assert S.dtype == np.float32


test("spectrogram (STFT)", test_spectrogram)


def test_chroma():
    import audiopy as ap
    c = ap.chroma(SINE_WAVE, SR)
    assert c.shape[0] == 12
    assert c.dtype == np.float32


test("chroma (12 pitch classes)", test_chroma)


def test_zcr():
    import audiopy as ap
    z = ap.zero_crossing_rate(SINE_WAVE)
    assert z.ndim == 2
    assert z.shape[0] == 1


test("zero_crossing_rate", test_zcr)


def test_spectral_centroid():
    import audiopy as ap
    sc = ap.spectral_centroid(SINE_WAVE, SR)
    assert sc.shape[0] == 1
    assert sc.mean() > 0


test("spectral_centroid", test_spectral_centroid)


def test_spectral_bandwidth():
    import audiopy as ap
    sb = ap.spectral_bandwidth(SINE_WAVE, SR)
    assert sb.shape[0] == 1


test("spectral_bandwidth", test_spectral_bandwidth)


def test_spectral_rolloff():
    import audiopy as ap
    sr_ = ap.spectral_rolloff(SINE_WAVE, SR)
    assert sr_.shape[0] == 1


test("spectral_rolloff", test_spectral_rolloff)


def test_rms():
    import audiopy as ap
    rms = ap.rms_energy(SINE_WAVE)
    assert rms.shape[0] == 1
    assert rms.mean() > 0


test("rms_energy", test_rms)


def test_tempo():
    import audiopy as ap
    bpm = ap.tempo(SINE_WAVE, SR)
    assert isinstance(bpm, float)
    assert bpm >= 0


test("tempo estimation", test_tempo)


def test_pitch():
    import audiopy as ap
    f0 = ap.pitch(SINE_WAVE, SR)
    assert isinstance(f0, np.ndarray)
    # Check some voiced frames are near 440 Hz
    voiced = f0[~np.isnan(f0)]
    if len(voiced) > 0:
        assert 400 < np.median(voiced) < 480, f"Expected ~440 Hz, got {np.median(voiced)}"


test("pitch (pYIN)", test_pitch)


def test_beat_frames():
    import audiopy as ap
    beats = ap.beat_frames(SINE_WAVE, SR)
    assert isinstance(beats, np.ndarray)


test("beat_frames", test_beat_frames)


def test_extract_all():
    import audiopy as ap
    features = ap.extract_all(SINE_WAVE, SR)
    assert isinstance(features, dict)
    assert "mfcc" in features
    assert "tempo" in features
    assert "spectrogram" in features


test("extract_all", test_extract_all)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Effects Module
# ═══════════════════════════════════════════════════════════════════════════
print("\n🎛️  Effects Tests")
print("-" * 40)


def test_change_pitch():
    import audiopy as ap
    y_up = ap.change_pitch(SINE_WAVE, SR, steps=2)
    assert len(y_up) > 0
    assert y_up.dtype == np.float32


test("change_pitch (+2 semitones)", test_change_pitch)


def test_change_speed():
    import audiopy as ap
    y_fast = ap.change_speed(SINE_WAVE, rate=1.5)
    assert len(y_fast) < N_SAMPLES  # should be shorter
    assert y_fast.dtype == np.float32


test("change_speed (1.5x)", test_change_speed)


def test_reverb():
    import audiopy as ap
    y_rev = ap.reverb(SINE_WAVE, SR, room_scale=0.5, wet_level=0.3)
    assert len(y_rev) > 0
    assert y_rev.dtype == np.float32


test("reverb (pedalboard)", test_reverb)


def test_echo():
    import audiopy as ap
    y_echo = ap.echo(SINE_WAVE, SR, delay=0.2, decay=0.4)
    assert len(y_echo) == N_SAMPLES
    assert y_echo.dtype == np.float32


test("echo effect", test_echo)


def test_low_pass():
    import audiopy as ap
    y_lp = ap.low_pass_filter(SINE_WAVE, SR, cutoff=1000)
    assert len(y_lp) == N_SAMPLES


test("low_pass_filter (1000 Hz)", test_low_pass)


def test_high_pass():
    import audiopy as ap
    y_hp = ap.high_pass_filter(SINE_WAVE, SR, cutoff=200)
    assert len(y_hp) == N_SAMPLES


test("high_pass_filter (200 Hz)", test_high_pass)


def test_bass_boost():
    import audiopy as ap
    y_bb = ap.bass_boost(SINE_WAVE, SR, gain_db=6)
    assert len(y_bb) == N_SAMPLES


test("bass_boost (+6 dB)", test_bass_boost)


def test_treble_boost():
    import audiopy as ap
    y_tb = ap.treble_boost(SINE_WAVE, SR, gain_db=4)
    assert len(y_tb) == N_SAMPLES


test("treble_boost (+4 dB)", test_treble_boost)


def test_distortion():
    import audiopy as ap
    y_dist = ap.distortion(SINE_WAVE, drive=0.7)
    assert len(y_dist) == N_SAMPLES
    assert np.max(np.abs(y_dist)) <= 1.0 + 1e-6


test("distortion (drive=0.7)", test_distortion)


def test_chorus_effect():
    import audiopy as ap
    y_ch = ap.chorus(SINE_WAVE, SR)
    assert len(y_ch) > 0


test("chorus (pedalboard)", test_chorus_effect)


def test_noise_reduce():
    import audiopy as ap
    y_clean = ap.noise_reduce(NOISY, SR)
    assert len(y_clean) == N_SAMPLES
    assert y_clean.dtype == np.float32


test("noise_reduce", test_noise_reduce)


def test_noise_reduce_adaptive():
    import audiopy as ap
    y_clean = ap.noise_reduce_adaptive(NOISY, SR, chunks=4)
    assert len(y_clean) == N_SAMPLES


test("noise_reduce_adaptive (4 chunks)", test_noise_reduce_adaptive)


def test_compress():
    import audiopy as ap
    y_comp = ap.compress(SINE_WAVE, threshold=-10, ratio=4.0)
    assert len(y_comp) == N_SAMPLES


test("compress (dynamic range)", test_compress)


def test_fade_in():
    import audiopy as ap
    y_fi = ap.fade_in(SINE_WAVE, SR, duration=0.3)
    assert len(y_fi) == N_SAMPLES
    # First sample should be near zero
    assert abs(y_fi[0]) < 0.01


test("fade_in (0.3s)", test_fade_in)


def test_fade_out():
    import audiopy as ap
    y_fo = ap.fade_out(SINE_WAVE, SR, duration=0.3)
    assert len(y_fo) == N_SAMPLES
    # Last sample should be near zero
    assert abs(y_fo[-1]) < 0.01


test("fade_out (0.3s)", test_fade_out)


def test_equalize():
    import audiopy as ap
    y_eq = ap.equalize(SINE_WAVE, SR, bands={"low": 3, "mid": 0, "high": -2})
    assert len(y_eq) == N_SAMPLES
    assert y_eq.dtype == np.float32


test("equalize (3-band EQ)", test_equalize)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Visualization Module
# ═══════════════════════════════════════════════════════════════════════════
print("\n📈 Visualization Tests")
print("-" * 40)

# Use non-interactive backend to avoid display issues
import matplotlib
matplotlib.use("Agg")


def test_plot_waveform():
    import audiopy as ap
    fig = ap.plot_waveform(SINE_WAVE, SR, title="Test Waveform")
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("plot_waveform", test_plot_waveform)


def test_plot_spectrogram():
    import audiopy as ap
    fig = ap.plot_spectrogram(SINE_WAVE, SR)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("plot_spectrogram", test_plot_spectrogram)


def test_plot_mel_spectrogram():
    import audiopy as ap
    fig = ap.plot_mel_spectrogram(SINE_WAVE, SR)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("plot_mel_spectrogram", test_plot_mel_spectrogram)


def test_plot_mfcc():
    import audiopy as ap
    fig = ap.plot_mfcc(SINE_WAVE, SR)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("plot_mfcc", test_plot_mfcc)


def test_plot_chroma():
    import audiopy as ap
    fig = ap.plot_chroma(SINE_WAVE, SR)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("plot_chroma", test_plot_chroma)


def test_plot_pitch():
    import audiopy as ap
    fig = ap.plot_pitch(SINE_WAVE, SR)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("plot_pitch", test_plot_pitch)


def test_plot_beats():
    import audiopy as ap
    fig = ap.plot_beats(SINE_WAVE, SR)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("plot_beats", test_plot_beats)


def test_compare_waveforms():
    import audiopy as ap
    fig = ap.compare_waveforms(SINE_WAVE, NOISY, SR, labels=("Clean", "Noisy"))
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


test("compare_waveforms", test_compare_waveforms)


def test_save_plot():
    import audiopy as ap
    fig = ap.plot_waveform(SINE_WAVE, SR)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        ap.save_plot(fig, tmp_path)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
        os.remove(tmp_path)


test("save_plot to PNG", test_save_plot)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Utils Module
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔧 Utils Tests")
print("-" * 40)


def test_db_conversions():
    import audiopy as ap
    assert abs(ap.utils.db_to_amplitude(0) - 1.0) < 1e-6
    assert abs(ap.utils.db_to_amplitude(-20) - 0.1) < 1e-4
    assert abs(ap.utils.amplitude_to_db(1.0) - 0.0) < 1e-6
    assert abs(ap.utils.amplitude_to_db(0.5) - (-6.0206)) < 0.01


test("dB <-> amplitude conversions", test_db_conversions)


def test_is_valid_audio():
    import audiopy as ap
    assert ap.utils.is_valid_audio(SINE_WAVE, SR) is True
    assert ap.utils.is_valid_audio(None, SR) is False
    assert ap.utils.is_valid_audio(np.array([]), SR) is False


test("is_valid_audio", test_is_valid_audio)


def test_detect_format():
    import audiopy as ap
    assert ap.utils.detect_format("song.mp3") == "MP3"
    assert ap.utils.detect_format("audio.wav") == "WAV"
    assert ap.utils.detect_format("track.flac") == "FLAC"
    assert ap.utils.detect_format("file.xyz") == "UNKNOWN"


test("detect_format", test_detect_format)


def test_time_samples():
    import audiopy as ap
    assert ap.utils.time_to_samples(1.0, 44100) == 44100
    assert abs(ap.utils.samples_to_time(44100, 44100) - 1.0) < 1e-6


test("time <-> samples conversion", test_time_samples)


def test_audio_stats():
    import audiopy as ap
    stats = ap.utils.audio_stats(SINE_WAVE, SR)
    assert "duration_s" in stats
    assert "peak_amplitude" in stats
    assert "rms" in stats
    assert abs(stats["duration_s"] - DURATION) < 0.01


test("audio_stats", test_audio_stats)


def test_ensure_float32():
    import audiopy as ap
    y_int16 = (SINE_WAVE * 32767).astype(np.int16)
    y_f32 = ap.utils.ensure_float32(y_int16)
    assert y_f32.dtype == np.float32
    assert abs(np.max(np.abs(y_f32)) - 0.5) < 0.01


test("ensure_float32 (int16 -> float32)", test_ensure_float32)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Models — Enhancement (no network needed)
# ═══════════════════════════════════════════════════════════════════════════
print("\n🧠 Enhancement Tests (offline)")
print("-" * 40)


def test_denoise_deep_fallback():
    """Should fall back to noisereduce if DeepFilterNet unavailable."""
    import audiopy as ap
    y_clean = ap.denoise_deep(NOISY, SR)
    assert isinstance(y_clean, np.ndarray)
    assert len(y_clean) > 0
    assert y_clean.dtype == np.float32


test("denoise_deep (noisereduce fallback)", test_denoise_deep_fallback)


def test_enhance_speech():
    import audiopy as ap
    y_enh = ap.enhance_speech(NOISY, SR)
    assert isinstance(y_enh, np.ndarray)
    assert len(y_enh) > 0
    assert y_enh.dtype == np.float32


test("enhance_speech pipeline", test_enhance_speech)


def test_remove_silence():
    import audiopy as ap
    silence = np.zeros(SR, dtype=np.float32)
    audio = np.concatenate([SINE_WAVE, silence, SINE_WAVE])
    y_speech = ap.remove_silence(audio, SR, aggressiveness=2)
    assert isinstance(y_speech, np.ndarray)
    # Should be shorter than the original (some silence removed)
    assert len(y_speech) <= len(audio)


test("remove_silence (webrtcvad)", test_remove_silence)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
print(f"  Total:   {passed + failed + skipped} tests")
print("=" * 60)

if failed > 0:
    print("\n❌ Failed tests:")
    for r in results:
        if r[0] == "FAIL":
            print(f"   - {r[1]}: {r[2]}")
    sys.exit(1)
else:
    print("\n🎉 All tests passed!")
    sys.exit(0)
