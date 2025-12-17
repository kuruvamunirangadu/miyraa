from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.api.main import app


@patch('src.api.main.get_engine')
def test_fingerprint_empty_text(mock_get_engine):
    dummy_engine = MagicMock()
    dummy_engine.predict.return_value = {
        "text": "",
        "embed": [0.0],
        "emotion": "neutral",
        "emotion_scores": {"neutral": 1.0},
        "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        "safety": {"blocked": False},
    }
    mock_get_engine.return_value = dummy_engine

    with TestClient(app) as client:
        response = client.post("/nlp/emotion/fingerprint", json={"text": ""})

    # Endpoint may reject empty input; ensure it returns a structured response
    assert response.status_code in (200, 400, 422)
    if response.status_code == 200:
        payload = response.json()
        assert "text" in payload
        assert payload["text"] == ""


@patch('src.api.main.get_engine')
def test_fingerprint_long_text(mock_get_engine):
    dummy_engine = MagicMock()
    dummy_engine.predict.return_value = {
        "text": "x" * 128,
        "embed": [0.0, 0.1],
        "emotion": "neutral",
        "emotion_scores": {"neutral": 1.0},
        "vad": {"valence": 0.1, "arousal": 0.1, "dominance": 0.1},
        "safety": {"blocked": False},
    }
    mock_get_engine.return_value = dummy_engine

    with TestClient(app) as client:
        response = client.post("/nlp/emotion/fingerprint", json={"text": "x" * 10000})

    assert response.status_code in (200, 400, 422, 413)
    if response.status_code == 200:
        payload = response.json()
        assert "embed" in payload
