import pytest

from efforttts.data.inventory import classify


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("rainbow_01_loud", ("reading", "loud", "rainbow_01")),
        ("rainbow_08_highpitch", ("reading", "highpitch", "rainbow_08")),
        ("sentences_05_loud", ("reading", "loud", "sentences_05")),
        ("sentences_24_regular", ("reading", "regular", "sentences_24")),
        ("emo_cuteness_sentences", ("emotion", "cuteness", None)),
        ("emo_embarassment_sentences", ("emotion", "embarassment", None)),
        ("freeform_speech_01", ("freeform", None, None)),
        ("interjection_agreement", ("interjection", "agreement", None)),
        ("something_else", ("other", None, None)),
        ("emo_anger_freeform", ("emotion_freeform", "anger", None)),
    ],
)
def test_classify(stem, expected):
    assert classify(stem) == expected