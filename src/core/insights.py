# src/core/insights.py
import warnings
warnings.filterwarnings("ignore")

class InsightEngine:

    def explain_store(self, df_store, df_global):
        insights = []

        if df_store["conversion_rate"].mean() < df_global["conversion_rate"].mean():
            insights.append("Low conversion")

        if df_store["footfall"].mean() < df_global["footfall"].mean():
            insights.append("Low footfall")

        if df_store["markdown_rate"].mean() > df_global["markdown_rate"].mean():
            insights.append("High markdown")

        return insights