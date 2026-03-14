# Premier League Match Predictor

A machine learning system that predicts Premier League match outcomes — **Home Win**, **Draw**, or **Away Win** — using XGBoost and Random Forest classifiers.

The project covers the full data science workflow: data generation, feature engineering, model training, a REST API, and an interactive dashboard.

---

## What it does

Given any two Premier League teams, the system predicts the most likely outcome and returns the probabilities for each result. It uses features like:

- Rolling form over the last 5 matches
- Average goals scored and conceded
- Head-to-head win rate (last 5 meetings)
- Shot conversion rate
- Home advantage indicator

---

## Project structure

```
football_predict_ml/
├── data/
│   ├── raw/                  # Match data (CSV + JSON)
│   └── processed/            # Cleaned data and feature matrix
├── src/
│   ├── data_collection.py    # Generate or fetch match data
│   ├── data_cleaning.py      # Validate and normalize raw data
│   ├── feature_engineering.py  # Build ML features
│   ├── train_model.py        # Train and tune models
│   ├── evaluate_model.py     # Metrics and plots
│   ├── visualize.py          # Standalone charts
│   └── predict.py            # Inference class
├── api/
│   └── main.py               # FastAPI prediction service
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── models/                   # Saved models and evaluation plots
├── tests/                    # pytest test suite
├── run_pipeline.py           # Run the full pipeline in one command
├── docker-compose.yml
└── requirements.txt
```

---

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This single command runs all five steps in order:

1. **Data generation** — creates 300 simulated Premier League fixtures with realistic stats (goals, shots, possession) based on team strength ratings. If you have a `FOOTBALL_DATA_API_KEY` in a `.env` file it will fetch live data instead.
2. **Data cleaning** — validates types, removes duplicates, standardises team names.
3. **Feature engineering** — builds rolling stats, H2H records, and form metrics for each match.
4. **Model training** — trains Random Forest and XGBoost with time-series cross-validation and hyperparameter search.
5. **Evaluation** — generates confusion matrices, feature importance charts, and a model comparison, saved to `models/plots/`.

### 3. Open the dashboard

```bash
streamlit run dashboard/app.py
```

Then open **http://localhost:8501**

---

## Dashboard pages

| Page | What you'll find |
|---|---|
| **Overview** | Season stats: total fixtures, home/away win rates, goals per matchday, result distribution |
| **Predict** | Pick two teams, choose a model, and get win/draw/loss probabilities with recent form shown |
| **Team Stats** | Points, wins, clean sheets, goal averages, form trend, conversion rate per team |
| **Model Insights** | Confusion matrices, feature importance rankings, model comparison charts |
| **Table** | Simulated standings with Champions League, Europa, and relegation zones highlighted |

---

## REST API

Start the API server:

```bash
uvicorn api.main:app --reload
```

**Predict a match:**

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
  "home_win": 0.42,
  "draw": 0.25,
  "away_win": 0.33,
  "predicted_outcome": "Home Win",
  "model_used": "xgboost"
}
```

Interactive API docs: **http://localhost:8000/docs**

You can also send batch predictions to `POST /predict/batch`.

---

## Teams (2025-26 season)

The dataset reflects the current Premier League season, with the three promoted clubs from the 2024-25 Championship:

| Clubs | |
|---|---|
| Arsenal · Aston Villa · Bournemouth | Brentford · Brighton · Burnley |
| Chelsea · Crystal Palace · Everton | Fulham · Leeds United · Liverpool |
| Manchester City · Manchester United | Newcastle · Nottingham Forest |
| Sunderland · Tottenham · West Ham | Wolves |

---

## ML features

| Feature | Description |
|---|---|
| `home_form` / `away_form` | Points earned in last 5 matches |
| `home_avg_scored` / `away_avg_scored` | Rolling average goals scored |
| `home_avg_conceded` / `away_avg_conceded` | Rolling average goals conceded |
| `home_conversion_rate` | Shots on target ÷ shots taken |
| `form_diff` | Difference in rolling form (home − away) |
| `h2h_win_rate` | Home team's win rate in last 5 H2H meetings |
| `home_goal_diff` | Rolling goal difference |
| `home_advantage` | Binary flag (always 1 for home side) |

---

## Model performance

| Model | CV accuracy (typical) |
|---|---|
| XGBoost | 52 – 58% |
| Random Forest | 50 – 55% |

Football match prediction is an inherently hard problem — top bookmakers sit around 55-60%. These numbers are in a reasonable range given that the training data is synthetic.

---

## Docker

```bash
# Start everything (API on :8000, dashboard on :8501)
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

## Tech stack

Python 3.11 · pandas · scikit-learn · XGBoost · FastAPI · Streamlit · Plotly · Docker
