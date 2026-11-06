"""Small, testable helpers shared by build_showcase.py and app.py (numpy, scipy, pandas only)."""

import io
import warnings

import numpy as np
from scipy.io import wavfile

warnings.filterwarnings(
    "ignore", category=wavfile.WavFileWarning
)  # harmless: some WAVs carry a non-data chunk


def level_p95_db(x, sr):
    """Speech level: 95th percentile of 25 ms frame RMS, in dBFS (same measure as level_by_k.py)."""
    n = int(0.025 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    return float(20 * np.log10(np.percentile(np.sqrt((fr**2).mean(axis=1)) + 1e-9, 95)))


def normalize_level(x, sr, target_db=-24.0, peak=0.98):
    """Scale to a common speech level; never let the peak pass `peak`."""
    y = x * 10 ** ((target_db - level_p95_db(x, sr)) / 20)
    p = float(np.abs(y).max())
    if p > peak:
        y = y * (peak / p)
    return y.astype("float32")


def apply_gain_db(x, gain_db, peak=0.98):
    """Gain with a soft limiter: tanh saturation only when the peak would pass `peak`."""
    y = x * 10 ** (gain_db / 20)
    if float(np.abs(y).max()) > peak:
        y = peak * np.tanh(y / peak)
    return y.astype("float32")


def to_wav_bytes(x, sr):
    buf = io.BytesIO()
    wavfile.write(buf, sr, (np.clip(x, -1, 1) * 32767).astype("int16"))
    return buf.getvalue()


def read_wav(path):
    sr, x = wavfile.read(str(path))
    x = x.astype("float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 4:  # integer WAV
        x = x / 32768.0
    return sr, x


def ramp_gain(setting, ref, high, max_gain_db):
    """Loudness ramp used in the app: 0 dB at the reference setting, rising linearly to max_gain_db at `high`, constant above."""
    if high <= ref:
        return 0.0
    return float(max_gain_db * min(max(setting - ref, 0.0) / (high - ref), 1.0))


def passes(row, max_audio=10.0, max_wer=0.2, min_sim=0.80, min_wps=1.6, max_dup=1.15):
    """Presentability filter (the same for every model). Returns (ok, reason)."""
    if not (row.get("audio_s", 0) <= max_audio):
        return False, f"longer than {max_audio:g} s (did not stop)"
    wer, sim = row.get("wer"), row.get("sim_to_centroid")
    if wer is None or sim is None or np.isnan(wer) or np.isnan(sim):
        return False, "no ASR or similarity score (run Step A)"
    if wer > max_wer:
        return False, f"WER above {max_wer:g}"
    if sim < min_sim:
        return False, f"similarity below {min_sim:g}"
    if row.get("words_per_s", 9) < min_wps:
        return False, f"slower than {min_wps:g} words/s"
    n_text = max(len(str(row.get("text", "")).split()), 1)
    if (
        len(
            str(
                row.get("hypothesis", "") if row.get("hypothesis") == row.get("hypothesis") else ""
            ).split()
        )
        / n_text
        >= max_dup
    ):
        return False, "repeated or duplicated words"
    return True, ""
