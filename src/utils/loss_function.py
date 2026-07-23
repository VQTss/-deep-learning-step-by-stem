import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y_true = np.asarray(y_true,dtype=float)
    y_pred = np.asarray(y_pred,dtype=float)
    
    if loss_type == "mse":
        loss_result = np.mean((y_true - y_pred)**2)
    elif loss_type == "bce":
        eps = 1e-15
        y_pred = np.clip(y_pred,eps,1-eps)
        loss_result = - np.mean(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif loss_type == "hinge":
        loss_result = np.mean(np.maximum(0, 1 - y_true * y_pred))
    elif loss_type == "cce":
        y_true = np.asarray(y_true, dtype=int)
        logits = y_pred
        n = len(y_true)
        losses = []
        for i in range(n):
            z = logits[i]
            max_z = np.max(z)
            logsumexp = max_z + np.log(np.sum(np.exp(z - max_z)))
            losses.append(-(z[y_true[i]] - logsumexp))
        loss_result = np.mean(losses)
    return float(round(loss_result, 4))


