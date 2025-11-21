from typing import Dict, Any


class DummyEngine:
    def predict(self, text: str) -> Dict[str, Any]:
        # Return a deterministic-ish fingerprint for testing
        return {
            "text": text,
            "embed": [0.0, 0.1, 0.2],
            "vad": {"v": 0.5, "a": 0.5, "d": 0.5},
            "emotions": {"joy": 0.1, "anger": 0.0},
            "safety": {"blocked": False},
        }


_ENGINE = DummyEngine()


def get_engine():
    return _ENGINE
