import numpy as np

def score(y_true: np.array, y_preds: np.array, threshold: int = 0.5):
    """
    y_true:  1-d array or list
    y_preds: 1-d array or list
    threshold: for soft scoring (lower for hard scoring)
    """
    y_true = np.array(y_true) 
    y_preds = np.array(y_preds) 
    assert y_true.ndim == 1, f"The y_true should have one dim, {y_true.ndim} was given" 
    hits = np.abs(y_preds - y_true < threshold) 
    return np.mean(hits)