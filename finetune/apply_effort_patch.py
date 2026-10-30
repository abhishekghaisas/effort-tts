#!/usr/bin/env python3
"""Patch src/model.py of chatterbox-finetuning for effort conditioning (v2, tolerant).

Run from the toolkit root:   python apply_effort_patch.py
Makes src/model.py.bak first. If anything does not match, nothing is changed and the relevant
lines of src/model.py are printed so they can be pasted back for a manual fix.

Edits to ChatterboxTrainerWrapper.forward:
  1. add an `effort=None` argument (per-clip scalar, shape [B]) at the end of the signature;
  2. feed it as emotion_adv instead of the hard-coded 0.5;
  3. when the wrapper's ref_conditioning flag is True (set by train.py), do not mask the first prompt-length target positions
     (the prompt now comes from a different clip; masking would remove the loss on the first
     6 s of every clip, which is the whole clip for most of them).
"""

import ast
import re
import shutil
import sys
from pathlib import Path

p = Path("src/model.py")
s = p.read_text()
if "effort=None" in s:
    print("src/model.py already patched.")
    sys.exit(0)

problems = []
new = s

# ---- 1. signature: find 'def forward(' and its matching ')' by paren counting -------------------
m = re.search(r"def forward\(", new)
if not m:
    problems.append("could not find 'def forward('")
else:
    i, depth = m.end(), 1
    while i < len(new) and depth:
        depth += {"(": 1, ")": -1}.get(new[i], 0)
        i += 1
    close = i - 1  # index of the matching ')'
    params = new[m.end() : close]
    if "effort" in params:
        problems.append("signature already mentions 'effort'")
    else:
        stripped = params.rstrip()
        ind_m = re.search(r"\n([ \t]+)\S", params)
        ind = ind_m.group(1) if ind_m else "        "
        sep = "" if stripped.endswith(",") or not stripped else ","
        new = new[: m.end()] + stripped + sep + f"\n{ind}effort=None" + new[close:]

# ---- 2. emotion_adv ------------------------------------------------------------------------------
pat = re.compile(
    r"\n([ \t]*)emotion_adv\s*=\s*0\.5\s*\*\s*torch\.ones\(\s*batch_size\s*,\s*1\s*,\s*1\s*\)(\.to\(device\))?"
)
m2 = pat.search(new)
if not m2:
    problems.append(
        "could not find the hard-coded 'emotion_adv = 0.5 * torch.ones(batch_size, 1, 1)' line"
    )
else:
    ind = m2.group(1)
    block = (
        f"\n{ind}if effort is None:\n{ind}    emotion_adv = 0.5 * torch.ones(batch_size, 1, 1).to(device)"
        f"\n{ind}else:\n{ind}    emotion_adv = effort.to(device).float().view(batch_size, 1, 1)"
    )
    new = new[: m2.start()] + block + new[m2.end() :]

# ---- 3. prompt masking ---------------------------------------------------------------------------
# Handles: 'actual_prompt_len = prompt_tokens.size(1)', and mask_prompt lines that compare against
# 'prompt_tokens.size(1)' or 'self.prompt_token_len'. The Turbo branch ('< prompt_lens[:, None]') is left alone.
GUARD = "getattr(self, 'ref_conditioning', False)"
m3 = re.search(r"\n([ \t]*)actual_prompt_len\s*=\s*prompt_tokens\.size\(1\)", new)
n_mask = 0
if m3:
    ind = m3.group(1)
    new = (
        new[: m3.start()]
        + f"\n{ind}actual_prompt_len = 0 if {GUARD} else prompt_tokens.size(1)"
        + new[m3.end() :]
    )
    n_mask += 1
pat3 = re.compile(
    r"(\n[ \t]*mask_prompt\s*=\s*torch\.arange\([^\n]*?<\s*)(prompt_tokens\.size\(1\)|self\.prompt_token_len)(?=[ \t]*(#[^\n]*)?\n)"
)
new, k = pat3.subn(lambda m: f"{m.group(1)}(0 if {GUARD} else {m.group(2)})", new)
n_mask += k
if n_mask == 0:
    problems.append(
        "could not find the prompt masking ('actual_prompt_len = prompt_tokens.size(1)' or "
        "'mask_prompt = ... < prompt_tokens.size(1)' / '< self.prompt_token_len')"
    )

if not problems:
    try:
        ast.parse(new)
    except SyntaxError as e:
        problems.append(f"patched file would not parse: {e}")

if problems:
    print("PATCH FAILED, nothing was changed:")
    for q in problems:
        print("  -", q)
    lines = s.splitlines()
    print("\nRelevant lines of src/model.py (paste this back):")
    for n, l in enumerate(lines, 1):
        if re.search(r"def forward|prompt|emotion_adv|speaker_emb|mask_", l):
            print(f"{n:4d}: {l}")
    sys.exit(1)

shutil.copy(p, "src/model.py.bak")
p.write_text(new)
print("Patched src/model.py (backup: src/model.py.bak).")
print(
    "  forward(..., effort=None); emotion_adv from effort; no prompt masking when wrapper.ref_conditioning is True"
)
