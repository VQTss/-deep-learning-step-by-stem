import numpy as np


def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual_sum = np.sum(np.power((y_true -  y_pred),2))
    mean_y = np.mean(y_true)
    total_sum = np.sum(np.power((y_true -  mean_y),2))
    if total_sum == 0:
        return  1.0 if residual_sum == 0 else 0.0
    score = 1 - (residual_sum / total_sum)
    return round(float(score),4)

def mae(y_true, y_pred) -> float:
    """
    Compute Mean Absolute Error (MAE) for 1D regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score = np.mean(np.abs(y_true - y_pred))
    return round(float(score), 4)

def rmse(y_true, y_pred) -> float:
    """
    Compute Root Mean Square Error (RMSE) for 1D regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score = np.sqrt(np.mean(np.power((y_true - y_pred), 2)))
    return round(float(score), 4)