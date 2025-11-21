from src.api.main import fingerprint, TextIn


def test_fingerprint_empty_text():
    payload = TextIn(text="")
    out = fingerprint(payload)
    assert "text" in out


def test_fingerprint_long_text():
    payload = TextIn(text="x" * 10000)
    out = fingerprint(payload)
    assert "embed" in out
