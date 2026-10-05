import csv

import numpy as np
import pytest

from efforttts.baselines import chatterbox_gen as G
from efforttts.baselines.references import best_window
from efforttts.baselines.test_script import load_script


def test_build_jobs_counts_and_uniqueness():
    lines = load_script()
    jobs = G.build_jobs(["p023", "p045"], lines, [0.5, 1.2], [0, 1])
    assert len(jobs) == 2 * 30 * 2 * 2
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert jobs[0]["job_id"] == "p023/C01_ex0.50_s0"
    # seeds nest outermost, so extending --seeds only appends new jobs
    first_pass = G.build_jobs(["p023", "p045"], lines, [0.5, 1.2], [0])
    assert [j["job_id"] for j in jobs[: len(first_pass)]] == [j["job_id"] for j in first_pass]


def test_done_ids_requires_ok_and_existing_file(tmp_path):
    (tmp_path / "p023").mkdir()
    (tmp_path / "p023" / "a.wav").write_bytes(b"x")
    rows = [
        {"job_id": "j1", "path": "p023/a.wav", "status": "ok"},
        {"job_id": "j2", "path": "p023/missing.wav", "status": "ok"},
        {"job_id": "j3", "path": "p023/a.wav", "status": "error: boom"},
    ]
    with open(tmp_path / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "path", "status"])
        w.writeheader()
        w.writerows(rows)
    assert G.done_ids(tmp_path / "results.csv", tmp_path) == {"j1"}
    assert G.done_ids(tmp_path / "nope.csv", tmp_path) == set()


def test_best_window_prefers_dense_speech():
    sr = 16000
    rng = np.random.default_rng(0)
    x = 0.001 * rng.standard_normal(sr * 40)          # near-silence
    t = np.arange(sr * 12) / sr
    x[sr * 22 : sr * 34] += 0.3 * np.sin(2 * np.pi * 150 * t)  # 12 s of continuous "speech"
    x[sr * 3 : sr * 4] += 0.3 * np.sin(2 * np.pi * 150 * t[:sr])  # short decoy burst
    start, dens = best_window(x, sr)
    assert 21.5 <= start <= 24.5 and dens > 0.9
    with pytest.raises(ValueError):
        best_window(x[: sr * 5], sr)


def test_dry_run_reports_without_torch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EFFORT_DATA_DIR", str(tmp_path))
    G.main(["--speakers", "23", "45", "--dry-run", "--seeds", "2"])
    out = capsys.readouterr().out
    assert "600 total jobs" in out and "600 pending" in out #pending and total job counts