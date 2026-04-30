# src/core/insights.py

import warnings

# Suppress non-critical warnings for cleaner logs
warnings.filterwarnings("ignore")


class InsightEngine:
    """
    Generates simple rule-based insights to explain store performance.
    """

    def explain_store(self, df_store, df_global):
        """
        Compare store metrics against global averages
        and return high-level performance insights.
        """
        insights = []

        # Check if conversion rate is below global average
        if df_store["conversion_rate"].mean() < df_global["conversion_rate"].mean():
            insights.append("Low conversion")

        # Check if footfall is below global average
        if df_store["footfall"].mean() < df_global["footfall"].mean():
            insights.append("Low footfall")

        # Check if markdown pressure is higher than global
        if df_store["markdown_rate"].mean() > df_global["markdown_rate"].mean():
            insights.append("High markdown")

        return insights