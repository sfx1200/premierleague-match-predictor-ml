"""
Full Pipeline Runner
====================
Executes the complete ML pipeline end-to-end:
  1. Data collection
  2. Data cleaning
  3. Feature engineering
  4. Model training
  5. Model evaluation
  6. Visualizations
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Create required directories
for d in ["data/raw", "data/processed", "models", "models/plots", "logs"]:
    Path(d).mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=" * 60)
    logger.info("Starting Premier League ML Pipeline")
    logger.info("=" * 60)

    # Step 1: Data collection
    logger.info("\n[1/5] Collecting match data...")
    from data_collection import collect_data
    df_raw = collect_data(use_synthetic=True)  # use synthetic unless API key set
    logger.info("  Raw matches collected: %d", len(df_raw))

    # Step 2: Data cleaning
    logger.info("\n[2/5] Cleaning data...")
    from data_cleaning import clean_data
    df_clean = clean_data()
    logger.info("  Clean matches: %d", len(df_clean))

    # Step 3: Feature engineering
    logger.info("\n[3/5] Engineering features...")
    from feature_engineering import engineer_features
    df_features = engineer_features()
    logger.info("  Feature matrix shape: %s", df_features.shape)

    # Step 4: Model training
    logger.info("\n[4/5] Training models...")
    from train_model import train
    results = train()
    for name, info in results.items():
        logger.info("  %s  CV accuracy: %.4f", name, info["cv_score"])

    # Step 5: Evaluation
    logger.info("\n[5/5] Evaluating models...")
    from evaluate_model import evaluate
    metrics = evaluate()
    for name, m in metrics.items():
        logger.info("  %s  Test accuracy: %.4f  F1: %.4f", name, m["accuracy"], m["f1"])

    # Bonus: Visualizations
    logger.info("\n[Bonus] Generating visualizations...")
    from visualize import generate_all_visuals
    generate_all_visuals()

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Models saved to:         models/")
    logger.info("  Plots saved to:          models/plots/")
    logger.info("  Processed data saved to: data/processed/")
    logger.info("")
    logger.info("To start the API:         uvicorn api.main:app --reload")
    logger.info("To start the dashboard:   streamlit run dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
