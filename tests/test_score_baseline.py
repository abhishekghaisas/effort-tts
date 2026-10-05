import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import wavfile

from efforttts import config
from efforttts.effort import estimator as E
from efforttts.eval import score_baseline as S
from efforttts.effort.features import SR as _SR  # noqa: F401  (module import check)

sys.path.insert(0, os.path.dirname(__file__))
from test_estimator import synth_features  # noqa: E402


def _tone(f0, rolloff, sr=24000, dur=2.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    x = sum(k ** (-rolloff) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, int(7000 / f0)))
    x = 0.3 * x / np.abs(x).max() + 1e-4 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFORT_DATA_DIR", str(tmp_path))
    synth_features().drop(columns=[E.F0_ST]).to_csv(config.processed_dir() / "features.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["estimator"])
    E.main()
    root = config.data_root() / "baselines" / "chatterbox" / "cfg0.50"
    root.mkdir(parents=True)
    rows, asr = [], []
    exs = [0.3, 0.5, 0.8, 1.2]
    rolloffs = {0.3: 2.2, 0.5: 2.0, 0.8: 1.2, 1.2: 0.9}   # flatter spectrum as "exaggeration" rises
    for spk in ("p013", "p014", "p015"):
        (root / spk).mkdir()
        for lid, intent in (("C01", "calm"), ("S01", "shout")):
            for ex in exs:
                rel = f"{spk}/{lid}_ex{ex:.2f}_s0.wav"
                wavfile.write(root / rel, 24000, _tone(150.0 + 30 * ex, rolloffs[ex]))
                jid = f"{spk}/{lid}_ex{ex:.2f}_s0"
                rows.append({"job_id": jid, "speaker": spk, "line_id": lid, "intent": intent,
                             "text": "Get back from the edge right now!", "exaggeration": ex,
                             "cfg_weight": 0.5, "seed": 0, "path": rel, "audio_s": 2.5, "gen_s": 5.0,
                             "status": "ok"})
                asr.append({"job_id": jid, "hypothesis": "get back from the edge", "wer": 0.1 + 0.1 * ex,
                            "sim_to_centroid": 0.9 - 0.1 * ex, "sim_to_reference": 0.95 - 0.1 * ex})
    pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    pd.DataFrame(asr).to_csv(root / "asr_speaker.csv", index=False)
    pd.DataFrame({"speaker": ["p013"] * 2, "stem": ["a", "b"], "category": ["reading"] * 2,
                  "label": ["regular", "loud"], "sim_to_centroid": [0.9, 0.8]}).to_csv(
        root / "real_calibration.csv", index=False)
    return root


def test_pipeline_runs_and_effort_rises_with_exaggeration(tmp_path, monkeypatch, capsys):
    root = _setup(tmp_path, monkeypatch)
    S.main(["--cfg-weight", "0.50"])
    df = pd.read_csv(root / "gap_clips.csv")
    assert len(df) == 3 * 2 * 4
    assert (root / "gap_report.png").exists()
    med = df.groupby("exaggeration").score_nolevel.median()
    assert med[1.2] > med[0.3]
    # paired delta is zero at the default exaggeration and positive for the top setting
    assert np.allclose(df[np.isclose(df.exaggeration, 0.5)].delta_nolevel, 0.0)
    assert df[np.isclose(df.exaggeration, 1.2)].delta_nolevel.median() > 0
    # WER and similarity were merged from step A
    assert df.wer.notna().all() and df.sim_to_centroid.notna().all()
    out = capsys.readouterr().out
    assert "By exaggeration" in out and "intent x exaggeration" in out


def test_scores_are_gain_invariant(tmp_path, monkeypatch):
    """A quieter copy of the same audio must get the same level-free score."""
    root = _setup(tmp_path, monkeypatch)
    import joblib

    bundle = joblib.load(config.processed_dir() / "estimator" / "estimator.joblib")
    ears = E.load_features()
    base = S.speaker_baselines(ears)
    x, sr = S.read_wav(root / "p013" / "C01_ex1.20_s0.wav")
    rows = pd.DataFrame([{"speaker": "p013", **S.clip_features(x, sr)},
                         {"speaker": "p013", **S.clip_features(0.1 * x, sr)}])
    sc = S.score_clips(rows, base, bundle)
    assert np.isclose(sc.score_nolevel[0], sc.score_nolevel[1], atol=1e-6)