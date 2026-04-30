# src/core/feature_engineer.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress non-critical warnings for cleaner logs
warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Applies feature engineering and encoding to raw dataset.
    """

    def transform(self, df):
        """
        Create additional features:
        - time-based features
        - simulated operational features
        """
        # Convert date and extract time-based features
        df["Date"] = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        df["year"] = df["Date"].dt.year
        df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

        # Simulated operational features (not available in dataset)
        df["footfall"] = df["Customers"] + np.random.normal(0, 10, len(df))
        df["conversion_rate"] = df["Customers"] / (df["footfall"] + 1)
        df["staff_hours"] = np.random.randint(5, 20, len(df))
        df["markdown_rate"] = df["Promo"] * np.random.uniform(0.1, 0.5, len(df))

        return df

    def encode(self, df, cat_cols):
        """
        Encode categorical variables using LabelEncoder.
        """
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        return df