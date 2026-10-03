from collections import Counter

from efforttts.data.splits import _norm_id, make_splits


def _genders():
    g = {}
    for i in range(1, 108):
        g[f"p{i:03d}"] = "female" if i % 2 == 0 else "male"
    return g


def test_disjoint_and_sizes():
    split = make_splits(_genders(), n_test=10, n_val=5, seed=1)
    c = Counter(split.values())
    assert c == {"test": 10, "val": 5, "train": 92}
    assert len(split) == 107


def test_gender_balanced_test():
    g = _genders()
    split = make_splits(g, 10, 5, seed=1)
    test_g = Counter(g[s] for s, v in split.items() if v == "test")
    assert abs(test_g["male"] - test_g["female"]) <= 1


def test_deterministic_and_seed_sensitive():
    g = _genders()
    assert make_splits(g, 10, 5, 7) == make_splits(g, 10, 5, 7)
    assert make_splits(g, 10, 5, 7) != make_splits(g, 10, 5, 8)


def test_norm_id():
    assert _norm_id("p001") == "p001"
    assert _norm_id(7) == "p007"
    assert _norm_id("P107") == "p107"