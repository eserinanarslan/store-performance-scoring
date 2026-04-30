# src/core/data_loader.py

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class DataLoader:

    def __init__(self, store_path, train_path):
        self.store_path = store_path
        self.train_path = train_path

    def load_data(self):
        store_df = pd.read_csv(self.store_path)
        train_df = pd.read_csv(self.train_path)

        train_df = train_df.merge(store_df, on="Store", how="left")

        return train_df