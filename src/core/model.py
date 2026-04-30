# src/core/model.py

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:

    def train(self, X_train, y_train):
        model = LGBMRegressor(random_state=42)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0]
        }

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        return grid.best_estimator_