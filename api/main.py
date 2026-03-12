"""
FastAPI Prediction Service
===========================
Provides a REST API for predicting Premier League match outcomes.

Endpoints
---------
GET  /              – health check
GET  /teams         – list available teams
POST /predict       – predict a single match outcome
POST /predict/batch – predict multiple matches
GET  /models        – list available trained models
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Ensure src is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import get_predictor, MatchPredictor  # noqa: E402

logger = logging.getLogger("uvicorn.error")

PL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham",
    "West Ham", "Wolves",
]

AVAILABLE_MODELS = ["xgboost", "random_forest"]

# ── Application lifecycle ──────────────────────────────────────────────────────

_predictors: dict[str, MatchPredictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models at startup."""
    for name in AVAILABLE_MODELS:
        try:
            _predictors[name] = get_predictor(name)
            logger.info("Pre-loaded model: %s", name)
        except Exception as exc:
            logger.warning("Could not pre-load model '%s': %s", name, exc)
    yield
    _predictors.clear()


app = FastAPI(
    title="⚽ Premier League Match Predictor",
    description=(
        "Predict the outcome of Premier League 2025-26 matches "
        "using Machine Learning (Random Forest & XGBoost)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    model: Literal["xgboost", "random_forest"] = "xgboost"

    @field_validator("home_team", "away_team")
    @classmethod
    def validate_team(cls, v: str) -> str:
        if v not in PL_TEAMS:
            raise ValueError(
                f"Unknown team '{v}'. Valid teams: {PL_TEAMS}"
            )
        return v

    @field_validator("away_team")
    @classmethod
    def teams_differ(cls, v: str, info) -> str:
        if info.data.get("home_team") and v == info.data["home_team"]:
            raise ValueError("home_team and away_team must be different.")
        return v


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    predicted_outcome: str
    model_used: str


class BatchPredictRequest(BaseModel):
    matches: list[PredictRequest]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "Premier League Match Predictor",
        "version": "1.0.0",
    }


@app.get("/teams", tags=["Data"])
def list_teams():
    """Return all valid Premier League teams for the 2025-26 season."""
    return {"teams": PL_TEAMS, "count": len(PL_TEAMS)}


@app.get("/models", tags=["Models"])
def list_models():
    """Return available trained models."""
    loaded = list(_predictors.keys())
    return {"available": AVAILABLE_MODELS, "loaded": loaded}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_match(request: PredictRequest):
    """
    Predict the outcome of a Premier League match.

    Returns probabilities for Home Win, Draw, and Away Win
    along with the most likely outcome.
    """
    if request.model not in _predictors:
        try:
            _predictors[request.model] = get_predictor(request.model)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{request.model}' is not trained yet. "
                       "Run `python src/train_model.py` first.",
            )

    predictor = _predictors[request.model]
    try:
        result = predictor.predict(request.home_team, request.away_team)
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictResponse(**result)


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """Predict outcomes for multiple matches in a single request."""
    results = []
    for match in request.matches:
        resp = predict_match(match)
        results.append(resp)
    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
