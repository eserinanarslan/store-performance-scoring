# src/core/model.py

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import warnings

# Suppress non-critical warnings for cleaner logs
warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Handles model training and hyperparameter tuning.
    """

    def train(self, X_train, y_train):
        """
        Train a LightGBM regression model using GridSearchCV
        to find the best hyperparameters.
        """

        # Initialize base model
        model = LGBMRegressor(random_state=42)

        # Define hyperparameter search space
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0]
        }

        # Perform grid search with cross-validation
        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        # Train models and select the best one
        grid.fit(X_train, y_train)

        return grid.best_estimator_