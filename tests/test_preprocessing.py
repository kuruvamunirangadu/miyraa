from src.nlp.preprocessing import normalize_text


def test_normalize_basic():
    s = "  Hello   WORLD\n"
    out = normalize_text(s)
    assert out == "hello world"

    assert normalize_text(None) == ""
