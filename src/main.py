# src/main.py

import configparser
import os

from api.routes import create_app, set_data

from core.data_loader import DataLoader
from core.feature_engineer import FeatureEngineer
from core.model import ModelTrainer
from core.evaluator import Evaluator
from core.scorer import Scorer

from utils.plotter import Plotter

from sklearn.model_selection import train_test_split

import warnings

# Suppress non-critical warnings for cleaner logs
warnings.filterwarnings("ignore")


def load_config():
    """
    Load service configuration from config.ini.
    """
    config = configparser.ConfigParser()
    config_path = os.path.join(os.getcwd(), "config.ini")

    if not config.read(config_path):
        raise FileNotFoundError(f"config.ini not found at {config_path}")

    host = config.get("Service", "host", fallback="0.0.0.0")
    port = config.getint("Service", "port", fallback=1001)

    return host, port


def run_pipeline():
    """
    Execute the full ML pipeline:
    data loading → feature engineering → training → scoring → outputs.
    """
    print("🚀 Starting data pipeline...")

    # =========================
    # 1. LOAD DATA
    # =========================
    loader = DataLoader(
        store_path="data/store.csv",
        train_path="data/train.csv"
    )
    train_df = loader.load_data()

    # =========================
    # 2. FEATURE ENGINEERING
    # =========================
    fe = FeatureEngineer()
    train_df = fe.transform(train_df)

    # Encode categorical variables
    cat_cols = ["StoreType", "Assortment", "StateHoliday"]
    train_df = fe.encode(train_df, cat_cols)

    # =========================
    # 3. FEATURES
    # =========================
    features = [
        "DayOfWeek", "Promo", "SchoolHoliday",
        "StoreType", "Assortment", "CompetitionDistance",
        "month", "year", "is_weekend",
        "footfall", "conversion_rate", "staff_hours", "markdown_rate"
    ]

    target = "Sales"

    X = train_df[features]
    y = train_df[target]

    # =========================
    # 4. TRAIN / VALIDATION SPLIT
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # 5. MODEL TRAINING
    # =========================
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)

    # =========================
    # 6. EVALUATION
    # =========================
    evaluator = Evaluator()
    metrics = evaluator.evaluate(model, X_val, y_val)

    print("\n📊 Model Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # =========================
    # 7. SCORING
    # =========================
    scorer = Scorer()
    train_df = scorer.score(train_df, model, features)

    # Assign performance labels based on score thresholds
    def classify(score):
        if score > 60:
            return "outperforming"
        elif score < 40:
            return "underperforming"
        else:
            return "normal"

    train_df["performance_label"] = train_df["score"].apply(classify)

    # Compute rolling trend per store
    train_df = train_df.sort_values(["Store", "Date"])
    train_df["rolling_score"] = train_df.groupby("Store")["score"].transform(
        lambda x: x.rolling(7).mean()
    )

    # =========================
    # 8. SAVE OUTPUT
    # =========================
    os.makedirs("outputs", exist_ok=True)

    # Save scored dataset for further analysis
    train_df.to_csv("outputs/scored_data.csv", index=False)
    print("💾 Saved: outputs/scored_data.csv")

    # =========================
    # 9. PLOTS
    # =========================
    print("📈 Generating plots...")

    plotter = Plotter(output_dir="outputs")

    plotter.plot_score_distribution(train_df)
    plotter.plot_label_distribution(train_df)
    plotter.plot_expected_vs_actual(train_df)

    # Select worst-performing store for detailed analysis
    worst_store = train_df.groupby("Store")["score"].mean().idxmin()
    df_store = train_df[train_df["Store"] == worst_store]

    plotter.plot_store_trend(train_df, worst_store)

    plotter.plot_store_vs_global(
        df_store,
        train_df,
        ["conversion_rate", "footfall", "staff_hours", "markdown_rate"],
        worst_store
    )

    print(f"📉 Worst store analyzed: {worst_store}")

    return train_df, model, features


def main():
    """
    Entry point:
    - load configuration
    - run pipeline
    - start API service
    """

    # Load configuration
    host, port = load_config()

    # Run pipeline once at startup
    train_df, model, features = run_pipeline()

    # Make data and model available to API
    set_data(train_df, model, features)

    print("🌐 Starting API...")

    app = create_app()
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()