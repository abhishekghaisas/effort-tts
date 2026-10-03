from efforttts.baselines.test_script import check_overlap, load_script, normalize, validate


def test_real_script_is_valid():
    lines = load_script()
    assert validate(lines) == []
    words = [len(normalize(l["text"])) for l in lines]
    assert min(words) <= 3 and max(words) >= 18  # varied length


def test_overlap_detects_planted_matches():
    lines = [{"id": "X1", "intent": "calm", "text": "The quick brown fox jumps over the dog."},
             {"id": "X2", "intent": "calm", "text": "Completely different words here."}]
    tr = {"a": "A quick brown fox jumps over a fence.", "b": "completely different words here"}
    hits = check_overlap(lines, tr)
    ids = {(h[0], h[1]) for h in hits}
    assert ("X1", "a") in ids and ("X2", "b") in ids
    assert not any(h[0] == "X1" and h[1] == "b" for h in hits)


def test_normalize():
    assert normalize("Don't STOP, now!") == ["don't", "stop", "now"]