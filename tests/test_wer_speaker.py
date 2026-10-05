import numpy as np

from efforttts.eval.asr_speaker import pick_centroid_windows, window_density
from efforttts.eval.wer import edit_distance, normalize_words, wer


def test_normalize_words_numbers_and_punctuation():
    assert normalize_words("Platform 6, at a quarter past 9!") == \
        ["platform", "six", "at", "a", "quarter", "past", "nine"]
    assert normalize_words("Room 12... starting NOW") == ["room", "twelve", "starting", "now"]
    assert normalize_words("It's twenty-one") == ["it's", "twenty", "one"]
    assert normalize_words("we\u2019re late") == ["we're", "late"]


def test_wer_basic_cases():
    assert wer("get back from the edge", "Get back from the edge!") == 0.0
    assert wer("a b c d", "a x c d") == 0.25
    assert wer("a b c d", "") == 1.0
    assert wer("a b", "a b c d") == 1.0          # insertions count
    assert wer("platform six", "platform 6") == 0.0
    assert wer("", "") == 0.0 and wer("", "x") == 1.0
    assert edit_distance(list("kitten"), list("sitting")) == 3


def _speechy(sr, seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    return 0.3 * np.sin(2 * np.pi * 150 * t) + 0.001 * rng.standard_normal(len(t))


def test_pick_centroid_windows_excludes_reference_and_spreads():
    sr = 16000
    audios = {"p001/a.wav": _speechy(sr, 100, 0), "p001/b.wav": _speechy(sr, 100, 1)}
    wins = pick_centroid_windows(audios, "p001/a.wav", 30.0, step_s=30.0, max_windows=4, sr=sr)
    assert len(wins) == 4
    assert {w[0] for w in wins} == {"p001/a.wav", "p001/b.wav"}      # spread across files
    for src, s in wins:
        if src == "p001/a.wav":
            assert not (s < 40.0 and s + 10.0 > 30.0)                # never overlaps the reference


def test_pick_centroid_windows_skips_silent_windows():
    sr = 16000
    x = np.concatenate([_speechy(sr, 20, 0), 0.0005 * np.random.default_rng(2).standard_normal(sr * 30)])
    assert window_density(x, sr, 0.0, 10.0) > 0.9
    wins = pick_centroid_windows({"s": x}, "none", 0.0, step_s=10.0, max_windows=10, sr=sr)
    assert all(s < 15.0 for _, s in wins)