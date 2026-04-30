import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Plotter:

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, name):
        path = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"📁 Saved: {path}")

    def plot_score_distribution(self, df):
        plt.figure(figsize=(8,5))
        sns.histplot(df["score"], bins=50)

        plt.axvline(50, linestyle="--")
        plt.axvline(40, linestyle="--")
        plt.axvline(60, linestyle="--")

        plt.title("Score Distribution")

        self._save("score_distribution")

    def plot_label_distribution(self, df):
        plt.figure(figsize=(6,4))
        df["performance_label"].value_counts().plot(kind="bar")
        plt.title("Performance Labels")

        self._save("label_distribution")

    def plot_expected_vs_actual(self, df):
        plt.figure(figsize=(8,6))

        sns.scatterplot(
            x=df["expected_sales"],
            y=df["Sales"],
            hue=df["performance_label"],
            alpha=0.3
        )

        max_val = df["Sales"].max()
        plt.plot([0, max_val], [0, max_val], linestyle="--")

        plt.title("Expected vs Actual")

        self._save("expected_vs_actual")

    def plot_store_trend(self, df, store_id):
        store_data = df[df["Store"] == store_id]

        plt.figure(figsize=(10,5))
        plt.plot(store_data["Date"], store_data["rolling_score"])

        plt.title(f"Store {store_id} Trend")

        self._save(f"store_{store_id}_trend")

    def plot_store_vs_global(self, df_store, df_global, features, store_id):
        comp = pd.DataFrame({
            "Store": df_store[features].mean(),
            "Global": df_global[features].mean()
        })

        ax = comp.plot(kind="bar")

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f"{height:.2f}",
                    ha='center',
                    va='bottom',
                    rotation=90
                )

        plt.title(f"Store {store_id} vs Global")

        self._save(f"store_{store_id}_comparison")