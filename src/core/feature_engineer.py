# src/core/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

class FeatureEngineer:

    def transform(self, df):
        df["Date"] = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        df["year"] = df["Date"].dt.year
        df["is_weekend"] = df["DayOfWeek"].isin([6,7]).astype(int)

        df["footfall"] = df["Customers"] + np.random.normal(0, 10, len(df))
        df["conversion_rate"] = df["Customers"] / (df["footfall"] + 1)
        df["staff_hours"] = np.random.randint(5, 20, len(df))
        df["markdown_rate"] = df["Promo"] * np.random.uniform(0.1, 0.5, len(df))

        return df

    def encode(self, df, cat_cols):
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df