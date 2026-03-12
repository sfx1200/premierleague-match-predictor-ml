# ⚽ Premier League 2025-26 Match Predictor

A production-ready Machine Learning system that predicts the outcome of Premier League matches (Home Win / Draw / Away Win) using Random Forest and XGBoost classifiers.

---

## Features

- **Full ML Pipeline** – data collection → cleaning → feature engineering → training → evaluation
- **Two Models** – Random Forest & XGBoost with hyperparameter tuning via TimeSeriesSplit CV
- **Advanced Features** – rolling form, H2H stats, conversion rates, offensive/defensive strength
- **REST API** – FastAPI service for real-time predictions
- **Interactive Dashboard** – Streamlit app for exploring stats and predictions
- **Dockerised** – single-command deployment with Docker Compose
- **Test Suite** – pytest tests for all pipeline stages

---

## Project Structure

```
football_predict_ml/
├── data/
│   ├── raw/               # Raw API / synthetic match data
│   └── processed/         # Cleaned data & feature matrix
├── src/
│   ├── data_collection.py   # Fetch PL data (live API or synthetic)
│   ├── data_cleaning.py     # Clean & validate raw data
│   ├── feature_engineering.py  # Build ML features
│   ├── train_model.py       # Train RF & XGBoost
│   ├── evaluate_model.py    # Metrics & plots
│   ├── visualize.py         # Standalone visualizations
│   └── predict.py           # Inference utility
├── api/
│   └── main.py              # FastAPI app
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── models/                  # Serialised models & plots
├── tests/                   # pytest test suite
├── run_pipeline.py          # One-command pipeline runner
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Set API key

```bash
cp .env.example .env
# Edit .env and add your football-data.org key
```

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

This will:
1. Generate / fetch match data
2. Clean the dataset
3. Engineer features (form, H2H, rolling stats, …)
4. Train both models with cross-validation & hyperparameter tuning
5. Evaluate and produce plots in `models/plots/`

---

## API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

### Predict a match

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Liverpool"}'
```

Response:

```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool",
  "home_win": 0.4231,
  "draw": 0.2487,
  "away_win": 0.3282,
  "predicted_outcome": "Home Win",
  "model_used": "xgboost"
}
```

Interactive docs: **http://localhost:8000/docs**

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501**

Pages:
- **Predict Match** – select teams and get probabilities
- **Team Stats** – form trends, goals, shots
- **Model Insights** – confusion matrices, feature importance
- **League Table** – simulated standings

---

## Docker

```bash
# Run everything (pipeline + API + dashboard)
docker-compose up

# API only
docker-compose up api
```

---

## Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## ML Features

| Feature | Description |
|---|---|
| `home/away_form` | Points from last 5 matches |
| `home/away_avg_scored` | Rolling avg goals scored (last 5) |
| `home/away_avg_conceded` | Rolling avg goals conceded (last 5) |
| `home/away_goal_diff` | Rolling goal difference |
| `home/away_conversion_rate` | Shots on target / shots |
| `form_diff` | Home form − Away form |
| `h2h_win_rate` | Home team H2H win rate (last 5 meetings) |
| `home_advantage` | Binary home indicator |

---

## Models

| Model | CV Accuracy (typical) |
|---|---|
| XGBoost | ~52-58% |
| Random Forest | ~50-55% |

> Note: Premier League prediction is inherently difficult; Vegas bookmakers achieve ~55-60%.

---

## Tech Stack

- **Python 3.11**
- pandas, numpy, scikit-learn, xgboost
- FastAPI, Uvicorn
- Streamlit, Plotly
- pytest, Docker

---

## License

MIT
