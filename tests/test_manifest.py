import math

import numpy as np

from efforttts.data.manifest import audio_stats


def test_audio_stats_levels():
    x = 0.5 * np.sin(np.linspace(0, 200 * math.pi, 48000)).astype("float32")
    s = audio_stats(x)
    assert abs(s["peak_dbfs"] - 20 * math.log10(0.5)) < 0.1
    assert abs(s["rms_dbfs"] - (20 * math.log10(0.5) - 3.01)) < 0.1
    assert s["clip_frac"] == 0.0


def test_audio_stats_clipping_and_empty():
    x = np.ones(100, dtype="float32")
    x[:10] = 0.2
    assert audio_stats(x)["clip_frac"] == 0.9
    assert audio_stats(np.array([]))["peak_dbfs"] == float("-inf")