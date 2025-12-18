from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app


@patch("src.api.main.get_engine")
def test_api_fingerprint(mock_get_engine):
    dummy_engine = MagicMock()
    dummy_engine.predict.return_value = {
        "text": "I am happy",
        "embed": [0.1, 0.2, 0.3],
        "emotion": "joy",
        "emotion_scores": {"joy": 0.9},
        "vad": {"valence": 0.8, "arousal": 0.4, "dominance": 0.6},
        "safety": {"blocked": False},
    }
    mock_get_engine.return_value = dummy_engine

    with TestClient(app) as client:
        response = client.post("/nlp/emotion/fingerprint", json={"text": "I am happy"})

    assert response.status_code == 200
    payload = response.json()
    assert "text" not in payload
    assert "processed_text" not in payload
    assert isinstance(payload.get("embed"), list)
    assert payload.get("pii_hashes") is not None
