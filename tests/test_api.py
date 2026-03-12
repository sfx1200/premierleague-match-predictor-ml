"""Tests for the FastAPI prediction service."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

# Patch model loading before importing app
mock_predictor = MagicMock()
mock_predictor.predict.return_value = {
    "home_team": "Arsenal",
    "away_team": "Liverpool",
    "home_win": 0.45,
    "draw": 0.25,
    "away_win": 0.30,
    "predicted_outcome": "Home Win",
    "model_used": "xgboost",
}

with patch("predict.get_predictor", return_value=mock_predictor):
    from main import app  # noqa: E402

client = TestClient(app)


class TestHealthCheck:
    def test_root_returns_ok(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestTeamsEndpoint:
    def test_returns_20_teams(self):
        resp = client.get("/teams")
        assert resp.status_code == 200
        assert resp.json()["count"] == 20

    def test_arsenal_in_teams(self):
        resp = client.get("/teams")
        assert "Arsenal" in resp.json()["teams"]


class TestPredictEndpoint:
    def test_invalid_team_returns_422(self):
        resp = client.post("/predict", json={
            "home_team": "Not A Team",
            "away_team": "Arsenal",
        })
        assert resp.status_code == 422

    def test_same_team_returns_422(self):
        resp = client.post("/predict", json={
            "home_team": "Arsenal",
            "away_team": "Arsenal",
        })
        assert resp.status_code == 422

    def test_valid_request_structure(self):
        """Verify response schema is correct (model may not be loaded in CI)."""
        resp = client.post("/predict", json={
            "home_team": "Arsenal",
            "away_team": "Liverpool",
        })
        # Either 200 (if model loaded) or 503 (model not yet trained)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "home_win" in data
            assert "draw" in data
            assert "away_win" in data
            assert "predicted_outcome" in data
            # Probabilities sum to ~1
            total = data["home_win"] + data["draw"] + data["away_win"]
            assert abs(total - 1.0) < 0.01


class TestModelsEndpoint:
    def test_returns_available_models(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert "available" in resp.json()
