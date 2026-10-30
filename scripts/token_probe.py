#!/usr/bin/env python3
"""Do the S3 speech tokens of LOUD reading differ from REGULAR reading? (small CPU-only probe)

Why: a T3-only fine-tune can only learn effort that is visible in the speech tokens it is trained to predict. If tokens of
loud and regular reads are statistically alike, effort lives in the (frozen) decoder and no T3 fine-tune can add it.

Step 1 (Chatterbox env, needs torch):
    python scripts/token_probe.py export --pre PATH/TO/EffortDataset/preprocess \
        --manifest EffortDataset/effort_manifest.jsonl --out tokens.npz
Step 2 (any env with scikit-learn):
    python scripts/token_probe.py analyze --tokens tokens.npz

Prints aggregates only (AUCs). A low AUC is NOT proof that tokens carry no effort (this probe only looks at token
frequencies and length), but a high AUC would show they do.
"""

import argparse
import json
import os

import numpy as np

VOCAB = 6561


def export(a):
    import torch

    toks, spk, lab, stem = [], [], [], []
    with open(a.manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r.get("label", "")).lower() not in ("regular", "loud"):
                continue
            p = os.path.join(a.pre, r["id"] + ".pt")
            if not os.path.exists(p):
                continue
            t = torch.load(p)["speech_tokens"].long().numpy()[:-1]  # drop the appended stop token
            toks.append(t)
            spk.append(r["speaker"])
            lab.append(int(r["label"].lower() == "loud"))
            stem.append(r["stem"])
    arr = np.empty(len(toks), dtype=object)
    for i, t in enumerate(toks):
        arr[i] = t
    np.savez(a.out, tokens=arr, speaker=np.array(spk), label=np.array(lab), stem=np.array(stem))
    print(f"exported {len(toks)} clips ({sum(lab)} loud) -> {a.out}")


def analyze(a):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    z = np.load(a.tokens, allow_pickle=True)
    toks, spk, y, stem = z["tokens"], z["speaker"], z["label"], z["stem"]
    n = len(y)
    length = np.array([len(t) for t in toks], dtype=float)
    rows = np.concatenate([np.full(len(t), i) for i, t in enumerate(toks)])
    cols = np.concatenate(toks)
    H = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, VOCAB)).toarray()
    H = H / length[:, None]
    Hc = H.copy()
    for s in np.unique(spk):  # speaker-centred: relative to that speaker's own clips
        m = spk == s
        Hc[m] -= H[m].mean(axis=0)
    ll = np.log(length)
    llc = ll.copy()
    for s in np.unique(spk):
        m = spk == s
        llc[m] -= ll[m].mean()
    feats = {
        "length only (speaker-centred)": llc[:, None],
        "token frequencies (raw)": H,
        "token frequencies (speaker-centred)": Hc,
    }
    print(
        f"{n} clips, {int(y.sum())} loud, {len(np.unique(spk))} speakers; grouped CV by speaker\n"
    )
    rng = np.random.default_rng(0)
    for name, X in feats.items():
        res, ctl = [], []
        for tr, te in GroupKFold(n_splits=min(4, len(np.unique(spk)))).split(X, y, spk):
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
            res.append(roc_auc_score(y[te], clf.decision_function(X[te])))
            ys = y[tr].copy()
            rng.shuffle(ys)
            ctl.append(
                roc_auc_score(
                    y[te],
                    LogisticRegression(C=1.0, max_iter=2000)
                    .fit(X[tr], ys)
                    .decision_function(X[te]),
                )
            )
        print(
            f"{name:42s} AUC {np.mean(res):.3f} +/- {np.std(res):.3f}   (shuffled-label control {np.mean(ctl):.3f})"
        )
    # paired duration effect on matched texts
    ratio = []
    for s in np.unique(spk):
        for st in set(x.rsplit("_", 1)[0] for x in stem[spk == s]):
            a_ = [length[i] for i in range(n) if spk[i] == s and stem[i] == st + "_regular"]
            b_ = [length[i] for i in range(n) if spk[i] == s and stem[i] == st + "_loud"]
            if a_ and b_:
                ratio.append(b_[0] / a_[0])
    if ratio:
        print(
            f"\nmatched texts: loud/regular token-length ratio median {np.median(ratio):.2f} (n={len(ratio)} pairs)"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--pre", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--out", default="tokens.npz")
    n_ = sub.add_parser("analyze")
    n_.add_argument("--tokens", default="tokens.npz")
    args = ap.parse_args()
    export(args) if args.cmd == "export" else analyze(args)
