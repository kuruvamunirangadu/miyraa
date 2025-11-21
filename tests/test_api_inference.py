from src.api.main import fingerprint, TextIn


def test_api_fingerprint():
    payload = TextIn(text="I am happy")
    j = fingerprint(payload)
    assert "text" in j and j["text"] == "I am happy"
    assert "embed" in j and isinstance(j["embed"], list)
