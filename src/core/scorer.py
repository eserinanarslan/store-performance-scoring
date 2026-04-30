# src/core/scorer.py
import warnings
warnings.filterwarnings("ignore")

class Scorer:

    def score(self, df, model, features):
        df["expected_sales"] = model.predict(df[features])
        df["performance_gap"] = df["Sales"] - df["expected_sales"]
        df["relative_gap"] = df["performance_gap"] / (df["expected_sales"] + 1)

        df["score"] = 50 + (df["relative_gap"] * 100)
        df["score"] = df["score"].clip(0, 100)

        return df