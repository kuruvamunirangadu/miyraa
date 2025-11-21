from src.nlp.preprocessing import normalize_text


def test_none_and_whitespace():
    assert normalize_text(None) == ""
    assert normalize_text("   ") == ""


def test_unicode_and_tabs():
    s = "\tCafé\n"
    assert normalize_text(s) == "café"


def test_no_lowercase():
    s = "Mixed CASE"
    assert normalize_text(s, lower=False) == "Mixed CASE"
