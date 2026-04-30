# src/core/scorer.py

import warnings

# Suppress non-critical warnings for cleaner logs
warnings.filterwarnings("ignore")


class Scorer:
    """
    Calculates store performance scores based on expected vs actual sales.
    """

    def score(self, df, model, features):
        """
        Generate expected sales using the trained model and compute:
        - performance gap
        - relative gap
        - final normalized score (0–100)
        """

        # Predict expected sales based on model
        df["expected_sales"] = model.predict(df[features])

        # Difference between actual and expected performance
        df["performance_gap"] = df["Sales"] - df["expected_sales"]

        # Normalize gap to make it comparable across stores
        df["relative_gap"] = df["performance_gap"] / (df["expected_sales"] + 1)

        # Convert to score centered around 50
        df["score"] = 50 + (df["relative_gap"] * 100)

        # Clip scores for readability and stability
        df["score"] = df["score"].clip(0, 100)

        return df