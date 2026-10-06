"""Inspect exaggeration-1.6 clips using only existing scores (read-only).

Run from the repo root in the project env:
    python inspect_exag16.py

Purpose: without per-clip listening labels, test whether the estimator and WER
already flag the problems heard by ear, and list a small set of clips to re-listen to.
Column names are guessed from gap_clips.csv; check the "Using columns" line and edit
the overrides below if a guess is wrong.
"""
import pandas as pd

CSV = "data_local/baselines/chatterbox/cfg0.50/gap_clips.csv"
DEFAULT_MEDIAN = -3.80   # median effort at exaggeration 0.5 (from the baseline table)
REAL_LOUD = 7.15         # real EARS loud, same estimator

# Overrides: set to a column name string if auto-detection picks the wrong one.
OVERRIDE = {"exag": "exaggeration", "effort": "score_nolevel", "wer": "wer",
            "intent": "intent", "speaker": "speaker", "sim": "sim_to_centroid",
            "file": "path"}


def find(df, keys, exclude=()):
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in keys) and not any(e in lc for e in exclude):
            return c
    return None


df = pd.read_csv(CSV)
cols = {
    "exag": OVERRIDE["exag"] or find(df, ["exagg"]),
    "effort": OVERRIDE["effort"] or find(df, ["effort"], exclude=["vs_default", "spectral", "nopitch"]),
    "wer": OVERRIDE["wer"] or find(df, ["wer"]),
    "intent": OVERRIDE["intent"] or find(df, ["intent"]),
    "speaker": OVERRIDE["speaker"] or find(df, ["speaker", "spk"]),
    "sim": OVERRIDE["sim"] or find(df, ["sim_centroid", "centroid"]),
    "file": OVERRIDE["file"] or find(df, ["path", "file", "wav", "clip", "id"]),
}
print("All columns:", df.columns.tolist())
print("Using columns:", cols)
missing = [k for k in ("exag", "effort", "wer", "intent") if cols[k] is None]
if missing:
    raise SystemExit(f"Could not find columns for {missing}; set OVERRIDE at the top.")

x = df[df[cols["exag"]].round(2) == 1.6].copy()
e, w, it = cols["effort"], cols["wer"], cols["intent"]
print(f"\n{len(x)} clips at exaggeration 1.6")

# 1. Is the WER mean driven by a few runaway clips?
top = x[w].sort_values(ascending=False)
n10 = max(1, int(0.1 * len(x)))
print("\n== WER at 1.6 ==")
print(f"mean {x[w].mean():.3f}, median {x[w].median():.3f}, max {x[w].max():.2f}")
print(f"fraction WER > 0.5: {(x[w] > 0.5).mean():.3f}   > 1.0: {(x[w] > 1.0).mean():.3f}")
print(f"mean without the worst 10% ({n10} clips): {top.iloc[n10:].mean():.3f}")

# 2. Does effort vary by intent, and is there a low-effort tail on shout lines?
print("\n== Effort quantiles at 1.6, by intent ==")
print(x.groupby(it)[e].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack().round(2))
print("\nShare of clips at 1.6 with effort <= default-setting median "
      f"({DEFAULT_MEDIAN}) [no more effortful than exaggeration 0.5]:")
print(x.groupby(it)[e].apply(lambda s: (s <= DEFAULT_MEDIAN).mean()).round(3))
print(f"\nShare with effort >= real loud ({REAL_LOUD}):")
print(x.groupby(it)[e].apply(lambda s: (s >= REAL_LOUD).mean()).round(3))

# 3. Clips to re-listen to (about 24 total)
show = [c for c in (cols["file"], cols["speaker"], it, e, w, cols["sim"], "audio_s")
        if c and c in df.columns]
def listing(title, d, with_text=False):
    print(f"\n== {title} ==")
    out = d[show].round(3).copy()
    if with_text and "hypothesis" in d.columns:
        out["text_words"] = d["text"].astype(str).str.split().str.len()
        out["heard_words"] = d["hypothesis"].astype(str).str.split().str.len()
        out["heard"] = d["hypothesis"].astype(str).str.slice(0, 70)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(out.to_string(index=False))

listing("Lowest-effort SHOUT lines (do they sound calm?)",
        x[x[it] == "shout"].nsmallest(8, e))
listing("Highest-effort CALM lines (do they sound needlessly loud?)",
        x[x[it] == "calm"].nlargest(8, e))
listing("Highest-WER clips (unintelligible? runaway?)", x.nlargest(8, w), with_text=True)