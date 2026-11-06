#!/usr/bin/env python3
"""Build the showcase folder (curated audio, manifest, metrics) from the scored roots. Run in the PROJECT env, repo root:
    python showcase/build_showcase.py --include-finetune

Needs, for every source root, the Step A and Step B outputs (gap_clips.csv with wer and sim_to_centroid). Sources (all 3 dev speakers x the 8 dev lines D01..D08):
  Qwen3-TTS steered, layer 21:  k = 0 (dev_base), 0.35 to 0.75 (dev2_L21), 1.0 to 1.5 (dev2_L21_trim2: trimmed to the end of the last word)
  Qwen3-TTS steered with the ANGER direction (optional, skipped if its roots are missing): k = 0.25 to 0.75 (dev2_L21_anger), 1.0 and 1.25 (dev2_L21_anger_trim)
  Chatterbox stock:             exaggeration 0.3 ... 1.6 (chatterbox_dev, made by showcase/gen_chatterbox_dev.py, then Step A and B with --default-ex 0.5)
  optional fine-tunes (the negative result): chatterbox_ft_v0 (effort_ft) and chatterbox_ft_v1 (effort_v1) on a few test lines
  metrics-only reference: the full stock Chatterbox baseline (5 speakers x 30 lines)
Curation: a clip is shown only if it passes ONE filter for every model (ends on its own within 10 s, WER <= 0.25, similarity >= 0.80, >= 1.6 words/s, no duplicated words; thresholds are
command-line options). A few removed clips per model and reason are also exported (audio_failed/) so the app can play examples of what the filter dropped.
Pass rates per model and setting are written to metrics.csv and shown in the app, so the curation is visible. Audio is level-normalised to a common speech level (-24 dBFS)
and written as 16-bit 24 kHz WAV. EARS-derived voices: the output folder must stay private until a publishing decision is made.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import showcase_lib as L

QWEN = [
    ("qwen_dev", "qwen_v2/dev_base", [0.0]),
    ("qwen_dev", "qwen_v2/dev2_L21", [0.35, 0.5, 0.6, 0.75]),
    ("qwen_dev", "qwen_v2/dev2_L21_trim2", [1.0, 1.25, 1.5]),
]
QWEN_ANGER = [
    ("qwen_anger", "qwen_v2/dev_base", [0.0]),
    ("qwen_anger", "qwen_v2/dev2_L21_anger", [0.25, 0.5, 0.75]),
    ("qwen_anger", "qwen_v2/dev2_L21_anger_trim", [1.0, 1.25]),
]
CB = [("chatterbox_dev", "chatterbox_dev/cfg0.50", [0.3, 0.5, 0.8, 1.2, 1.6])]
FT = [
    ("chatterbox_ft_v0", "effort_ft/cfg0.50", [0.5, 1.6, 2.0]),
    ("chatterbox_ft_v1", "effort_v1/cfg0.50", [0.5, 1.5, 3.0]),
]
MODEL_OF = {
    "qwen_dev": "Qwen3-TTS + steering",
    "qwen_anger": "Qwen3-TTS + steering (anger direction)",
    "chatterbox_dev": "Chatterbox (stock)",
    "chatterbox_ft_v0": "Chatterbox + LoRA v0",
    "chatterbox_ft_v1": "Chatterbox + LoRA v1",
}


def flag(group, s):
    if group == "qwen_dev":
        return (
            "ending trimmed; the voice starts to degrade"
            if s >= 1.5
            else "strong: can drift towards excited or manic delivery"
            if s >= 1.25
            else ""
        )
    if group == "qwen_anger":
        return (
            "strong: arousal stops rising and a few clips garble (the listener heard controlled urgency)"
            if s >= 1.25
            else ""
        )
    if group == "chatterbox_dev":
        return (
            "frequent speech problems and a large identity cost"
            if s >= 1.6
            else "identity and stability cost grows"
            if s >= 1.2
            else ""
        )
    return "fine-tune: the control barely changes the effort (negative result)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_local/baselines")
    ap.add_argument("--out", default="showcase")
    ap.add_argument("--speakers", nargs="*", default=["p023", "p045", "p056"])
    ap.add_argument("--dev-lines", default="qwen_v2/dev_lines.json")
    ap.add_argument("--ft-lines", nargs="*", default=["S03", "S08", "U01"])
    ap.add_argument("--target-db", type=float, default=-24.0)
    ap.add_argument("--include-finetune", action="store_true")
    ap.add_argument(
        "--max-wer",
        type=float,
        default=0.25,
        help="filter: maximum WER (Whisper is less reliable on shouted speech; 0.25 to 0.34 is more lenient on 6 to 9 word lines)",
    )
    ap.add_argument("--min-sim", type=float, default=0.80)
    ap.add_argument("--min-wps", type=float, default=1.6)
    ap.add_argument(
        "--failed-examples",
        type=int,
        default=2,
        help="export up to this many removed clips per model and reason, so visitors can hear what the filter dropped",
    )
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    dev_ids = [l["line_id"] for l in json.load(open(a.dev_lines))]
    sources = QWEN + QWEN_ANGER + CB + (FT if a.include_finetune else [])
    rows, log = [], []
    kw = dict(max_wer=a.max_wer, min_sim=a.min_sim, min_wps=a.min_wps)
    out.mkdir(parents=True, exist_ok=True)
    (out / "filter.json").write_text(
        json.dumps(
            {
                "max_audio_s": 10.0,
                "max_wer": a.max_wer,
                "min_sim": a.min_sim,
                "min_wps": a.min_wps,
                "max_dup": 1.15,
            }
        )
    )
    failed_n = {}
    for group, rel, settings in sources:
        root = base / rel
        if not (root / "gap_clips.csv").exists():
            log.append(f"SKIPPED {group} {rel}: no gap_clips.csv (run Step A and B)")
            continue
        g = pd.read_csv(root / "gap_clips.csv")
        g = g[
            g.speaker.isin(a.speakers)
            & g.exaggeration.apply(lambda e: any(np.isclose(e, s) for s in settings))
        ]
        g = g[g.line_id.isin(a.ft_lines if group.startswith("chatterbox_ft") else dev_ids)]
        emo = pd.read_csv(root / "emotion_dim.csv") if (root / "emotion_dim.csv").exists() else None
        for _, r in g.iterrows():
            ok, why = L.passes(r.to_dict(), **kw)
            row = {
                "group": group,
                "model": MODEL_OF[group],
                "speaker": r.speaker,
                "line_id": r.line_id,
                "intent": r.get("intent", ""),
                "text": r.text,
                "setting": float(r.exaggeration),
                "effort": r.get("score_nolevel", np.nan),
                "sim": r.get("sim_to_centroid", np.nan),
                "wer": r.get("wer", np.nan),
                "words_per_s": r.get("words_per_s", np.nan),
                "arousal": np.nan,
                "valence": np.nan,
                "shown": bool(ok),
                "reason": why,
                "flag": flag(group, float(r.exaggeration)),
                "audio": "",
                "failed_audio": "",
            }
            if emo is not None:
                m = emo[
                    (emo.speaker == r.speaker)
                    & (emo.line_id == r.line_id)
                    & np.isclose(emo.exaggeration, r.exaggeration)
                ]
                if len(m):
                    row["arousal"], row["valence"] = (
                        float(m.arousal.iloc[0]),
                        float(m.valence.iloc[0]),
                    )
            if ok:
                sr, x = L.read_wav(root / r["path"])
                rel_audio = (
                    Path("audio")
                    / group
                    / r.speaker
                    / f"{r.line_id}_{float(r.exaggeration):.2f}.wav"
                )
                (out / rel_audio).parent.mkdir(parents=True, exist_ok=True)
                (out / rel_audio).write_bytes(
                    L.to_wav_bytes(L.normalize_level(x, sr, a.target_db), sr)
                )
                row["audio"] = str(rel_audio)
            elif (
                not group.startswith("chatterbox_ft")
                and failed_n.get((group, why), 0) < a.failed_examples
            ):
                failed_n[(group, why)] = failed_n.get((group, why), 0) + 1
                sr, x = L.read_wav(root / r["path"])
                rel_failed = (
                    Path("audio_failed")
                    / group
                    / r.speaker
                    / f"{r.line_id}_{float(r.exaggeration):.2f}.wav"
                )
                (out / rel_failed).parent.mkdir(parents=True, exist_ok=True)
                (out / rel_failed).write_bytes(
                    L.to_wav_bytes(L.normalize_level(x, sr, a.target_db), sr)
                )
                row["failed_audio"] = str(rel_failed)
            rows.append(row)
        log.append(f"{group} {rel}: {len(g)} candidate clips")
    man = pd.DataFrame(rows)
    if man.empty:
        raise SystemExit("\n".join(log) + "\nNo clips found.")
    man.to_csv(out / "manifest.csv", index=False)

    def agg(d):
        return pd.Series(
            {
                "n_total": len(d),
                "n_shown": int(d.shown.sum()),
                "pass_rate": round(float(d.shown.mean()), 3),
                "effort_median": round(float(d.effort.median()), 2),
                "sim_median": round(float(d.sim.median()), 3),
                "wer_mean": round(float(d.wer.mean()), 3),
                "arousal_median": round(float(d.arousal.median()), 2)
                if d.arousal.notna().any()
                else np.nan,
                "valence_median": round(float(d.valence.median()), 2)
                if d.valence.notna().any()
                else np.nan,
            }
        )

    met = man.groupby(["group", "model", "setting"]).apply(agg).reset_index()
    full = base / "chatterbox/cfg0.50/gap_clips.csv"
    if full.exists():
        f = pd.read_csv(full)
        f["shown"] = [L.passes(r.to_dict(), **kw)[0] for _, r in f.iterrows()]
        fm = (
            f.groupby("exaggeration")
            .apply(
                lambda d: pd.Series(
                    {
                        "n_total": len(d),
                        "n_shown": int(d.shown.sum()),
                        "pass_rate": round(float(d.shown.mean()), 3),
                        "effort_median": round(float(d.score_nolevel.median()), 2),
                        "sim_median": round(float(d.sim_to_centroid.median()), 3),
                        "wer_mean": round(float(d.wer.mean()), 3),
                        "arousal_median": np.nan,
                        "valence_median": np.nan,
                    }
                )
            )
            .reset_index()
        )
        fm.insert(0, "model", "Chatterbox (stock), full baseline: 5 speakers x 30 lines")
        fm.insert(0, "group", "chatterbox_full")
        fm = fm.rename(columns={"exaggeration": "setting"})
        met = pd.concat([met, fm], ignore_index=True)
    met.to_csv(out / "metrics.csv", index=False)
    (out / "curation_log.txt").write_text(
        "\n".join(log)
        + "\n\n"
        + met.to_string(index=False)
        + "\n\nReasons clips were not shown:\n"
        + man[~man.shown].groupby(["group", "reason"]).size().to_string()
        + "\n"
    )
    print("\n".join(log))
    print(met.to_string(index=False))
    print(
        f"\n{int(man.shown.sum())} of {len(man)} clips shown; audio in {out / 'audio'} ({sum(f.stat().st_size for f in (out / 'audio').rglob('*.wav')) / 1e6:.0f} MB)"
    )


if __name__ == "__main__":
    main()
