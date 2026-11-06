#!/usr/bin/env python3
"""Remove a valence component from an effort direction (per layer). Research use; EARS-derived files stay local.

Idea (from the arousal / dominance / valence anchors): real ecstasy and real anger have about the same arousal (0.71 vs 0.70) and dominance (0.73 vs 0.71) but very different
valence (0.63 vs 0.24). So  valence_direction = direction(ecstasy) - direction(anger)  isolates valence at matched arousal. Projecting it out of the loud-reading effort direction
should keep the arousal gain and reduce the drift towards excited / manic delivery:   v_new = v - alpha * (v . u) u,  u = valence_direction / |valence_direction|.

Needs three direction files made by qwen_extract.py with the SAME settings (default loud reading, --loud-emotion anger, --loud-emotion extasy). Python env of qwen_extract.py (torch):
    python qwen_v2/orthogonalize_direction.py --effort qwen_v2/directions_06B.pt --neg qwen_v2/directions_06B_anger.pt \
        --pos qwen_v2/directions_06B_extasy.pt --out qwen_v2/directions_06B_noval.pt --alpha 1.0
Prints, per layer, the cosine between the effort direction and the valence direction and how much of the effort vector was removed.
"""

import argparse


def project_out(v, u, alpha=1.0):
    """v, u: vectors (numpy or torch). Returns v - alpha * (v . u) u / (u . u); v unchanged if u is all zeros."""
    uu = float(u @ u)
    return v if uu == 0 else v - alpha * float(v @ u) / uu * u


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", required=True)
    ap.add_argument(
        "--pos", required=True, help="direction file of the positive-valence emotion (extasy)"
    )
    ap.add_argument(
        "--neg",
        required=True,
        help="direction file of the negative-valence emotion at similar arousal (anger)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    eff, pos, neg = (torch.load(p, map_location="cpu") for p in (a.effort, a.pos, a.neg))
    out = {}
    print("layer  cos(effort, valence dir)  removed share of |effort|")
    for l, v in eff["layers"].items():
        u = pos["layers"][l] - neg["layers"][l]
        w = project_out(v, u, a.alpha)
        out[l] = w
        cos = float(v @ u) / (float(v.norm()) * float(u.norm()) + 1e-9)
        print(f"{l:5d}  {cos:+.3f}  {1 - float(w.norm()) / float(v.norm()):.3f}")
    torch.save(
        {
            "layers": out,
            "meta": {
                **eff.get("meta", {}),
                "orthogonalized": {"alpha": a.alpha, "pos": a.pos, "neg": a.neg},
            },
        },
        a.out,
    )
    print("saved", a.out)


if __name__ == "__main__":
    main()
