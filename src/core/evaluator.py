# src/core/evaluator.py

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")


class Evaluator:
    """
    Evaluates model performance using standard regression metrics.
    """

    def evaluate(self, model, X_val, y_val):
        """
        Generate predictions on validation data and compute metrics:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R2 (Coefficient of Determination)
        """
        # Predict on validation set
        pred = model.predict(X_val)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2 = r2_score(y_val, pred)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }