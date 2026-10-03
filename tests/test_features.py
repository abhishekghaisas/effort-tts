import numpy as np

from efforttts.effort.features import ALL_FEATURES, FEATURE_GROUPS, SR, extract_features


def harmonic(f0, rolloff, dur=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    x = sum(k ** (-rolloff) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, int(7000 / f0)))
    x = 0.3 * x / np.abs(x).max()
    return x + 1e-4 * rng.standard_normal(len(x))


def test_f0_and_voicing():
    f = extract_features(harmonic(190.0, 1.0), SR)
    assert abs(f["f0_median_hz"] - 190.0) / 190.0 < 0.02
    assert f["voiced_frac"] > 0.9


def test_noise_is_unvoiced():
    x = 0.1 * np.random.default_rng(1).standard_normal(SR)
    f = extract_features(x, SR)
    assert f["voiced_frac"] < 0.1
    assert np.isnan(f["f0_median_hz"])  # voiced-only features are NaN


def test_gain_invariance_except_level():
    x = harmonic(150.0, 1.5)
    a, b = extract_features(x, SR), extract_features(10 * x, SR)
    assert abs((b["level_active_db"] - a["level_active_db"]) - 20.0) < 0.01
    for name in ALL_FEATURES:
        if name in FEATURE_GROUPS["level"]:
            continue
        assert np.isclose(a[name], b[name], rtol=1e-5, atol=1e-5, equal_nan=True), name


def test_shallow_rolloff_raises_alpha_and_tilt():
    flat = extract_features(harmonic(200.0, 0.5), SR)
    steep = extract_features(harmonic(200.0, 3.0), SR)
    assert flat["alpha_ratio_db"] > steep["alpha_ratio_db"]
    assert flat["tilt_db_per_khz"] > steep["tilt_db_per_khz"]


def test_silence_and_short_input():
    assert np.isnan(extract_features(np.zeros(SR), SR)["level_active_db"])
    assert "level_active_db" in extract_features(np.ones(100) * 0.01, SR)


def _rumble_whisper(rumble_gain):
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(0)
    n = SR * 3
    band = sosfilt(butter(4, [1000, 4000], btype="band", fs=SR, output="sos"), rng.standard_normal(n))
    lowf = sosfilt(butter(4, 40, btype="low", fs=SR, output="sos"), rng.standard_normal(n))
    return 0.02 * band / band.std() + rumble_gain * lowf / lowf.std()


def test_rumble_under_whisper_is_not_voiced():
    # Regression: real whispered speech was reported ~87% voiced at ~930 Hz.
    for gain in (0.02, 0.1):
        f = extract_features(_rumble_whisper(gain), SR)
        assert f["voiced_frac"] < 0.1, gain


def test_no_octave_errors_at_high_f0():
    for f0 in (90.0, 300.0, 350.0, 450.0, 600.0):
        est = extract_features(harmonic(f0, 1.0), SR)["f0_median_hz"]
        assert abs(est - f0) / f0 < 0.02, (f0, est)


def test_remove_short_runs():
    from efforttts.effort.features import _remove_short_runs

    m = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    out = _remove_short_runs(m, 4)
    assert out.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
    assert _remove_short_runs(np.zeros(5, dtype=bool), 4).sum() == 0


def test_f0_gate_removes_far_outlier_bursts():
    # Steady 150 Hz voicing with a short 450 Hz burst in the middle: gate keeps the median at 150.
    a = harmonic(150.0, 1.0, dur=1.0, seed=1)
    b = harmonic(450.0, 1.0, dur=0.12, seed=2)
    x = np.concatenate([a[: SR // 2], b, a[SR // 2 :]])
    f = extract_features(x, SR)
    assert abs(f["f0_median_hz"] - 150.0) / 150.0 < 0.03
    assert f["f0_range_st"] < 3.0