# src/core/evaluator.py

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Evaluator:

    def evaluate(self, model, X_val, y_val):
        pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2 = r2_score(y_val, pred)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }