# src/core/data_loader.py

import pandas as pd
import warnings

# Suppress non-critical warnings for cleaner logs
warnings.filterwarnings("ignore")


class DataLoader:
    """
    Handles loading and merging of raw datasets.
    """

    def __init__(self, store_path, train_path):
        # File paths for input datasets
        self.store_path = store_path
        self.train_path = train_path

    def load_data(self):
        """
        Load store and training datasets and merge them
        on the Store identifier.
        """
        # Read input CSV files
        store_df = pd.read_csv(self.store_path)
        train_df = pd.read_csv(self.train_path)

        # Merge store metadata into training data
        train_df = train_df.merge(store_df, on="Store", how="left")

        return train_df