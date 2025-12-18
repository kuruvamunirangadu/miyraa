from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.nlp.safety import pii_scrub


@patch("src.api.main.scrub_pii")
@patch("src.api.main.get_engine")
def test_fingerprint_scrubs_pii_by_default(mock_get_engine, mock_scrub_pii):
    sanitized_text = "clean-text"
    metadata = {"method": "regex", "mappings": {"[EMAIL_deadbeef]": "deadbeef"}}
    mock_scrub_pii.return_value = (sanitized_text, metadata)

    dummy_engine = MagicMock()
    dummy_engine.predict.return_value = {"emotion": "joy", "processed_text": sanitized_text}
    mock_get_engine.return_value = dummy_engine

    with TestClient(app) as client:
        response = client.post("/nlp/emotion/fingerprint", json={"text": "raw@example.com"})

    assert response.status_code == 200
    body = response.json()

    mock_scrub_pii.assert_called_once_with("raw@example.com")
    dummy_engine.predict.assert_called_once_with("clean-text")
    assert body.get("pii_hashes") == metadata
    assert "text" not in body
    assert "processed_text" not in body


def test_scrub_pii_fallback_to_regex(monkeypatch):
    monkeypatch.setattr(pii_scrub, "PRESIDIO_AVAILABLE", False)
    text = "Reach me at user@example.com"

    scrubbed, meta = pii_scrub.scrub_pii(text, use_presidio=True)

    assert "[EMAIL" in scrubbed
    assert meta["method"] == "regex"
    assert meta["mappings"]


@patch("src.api.main.scrub_pii")
@patch("src.api.main.get_engine")
def test_logging_excludes_raw_text(mock_get_engine, mock_scrub_pii, caplog):
    sanitized_text = "safe"
    metadata = {"method": "regex", "mappings": {}}
    mock_scrub_pii.return_value = (sanitized_text, metadata)

    engine = MagicMock()
    engine.predict.return_value = {"emotion": "joy"}
    mock_get_engine.return_value = engine

    raw_text = "super secret text"
    with TestClient(app) as client:
        with caplog.at_level("INFO", logger="miyraa"):
            response = client.post("/nlp/emotion/fingerprint", json={"text": raw_text})

    assert response.status_code == 200
    for record in caplog.records:
        assert raw_text not in record.message
    