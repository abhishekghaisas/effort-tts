"""Word error rate with light normalization (lowercase, punctuation, small numbers)."""
from __future__ import annotations

import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _int_to_words(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    t, o = divmod(n, 10)
    return [_TENS[t]] + ([_ONES[o]] if o else [])


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation, spell out integers 0-99 (ASR often returns digits)."""
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    words: list[str] = []
    for w in t.split():
        if w.isdigit() and int(w) < 100:
            words += _int_to_words(int(w))
        else:
            w = w.strip("'")
            if w:
                words.append(w)
    return words


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = edit distance / reference length (can exceed 1 with insertions)."""
    r, h = normalize_words(reference), normalize_words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)