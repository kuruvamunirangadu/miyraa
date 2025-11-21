import pytest
from fastapi.testclient import TestClient

import src.server.app as appmod


def test_health_reflects_warmed_flag():
    """Smoke test: when the app's warmed flag is True, /health should report it.

    This test avoids triggering any heavyweight model downloads by directly
    setting the `app.state.warmed` flag. It's a small unit/integration smoke
    test to ensure the endpoint exposes the intended health information.
    """
    # Ensure deterministic state for the test
    appmod.app.state.warmed = True

    with TestClient(appmod.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("warmed") is True
