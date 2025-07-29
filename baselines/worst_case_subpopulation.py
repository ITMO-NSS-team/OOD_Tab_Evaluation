import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from typing import Union, Tuple, List


def estimate_conditional_risk(
    losses: np.ndarray,
    Z: Union[pd.DataFrame, np.ndarray],
    base_regressor=None
) -> GradientBoostingRegressor:
    """
    Fit h(Z) ≈ E[ loss | Z ] on provided sample.
    """
    if base_regressor is None:
        base_regressor = GradientBoostingRegressor(n_estimators=200, max_depth=3)
    model = clone(base_regressor)
    model.fit(Z, losses)
    return model


def compute_worst_cvar(
    h_vals: np.ndarray,
    alpha: float
) -> Tuple[float, float, np.ndarray]:
    """
    Find η minimizing η + (1/(α m)) Σ[max(h_vals - η, 0)],
    and return the corresponding subpopulation indices.

    Returns:
        W_alpha: worst-case subgroup loss estimate,
        eta_opt: optimal η threshold,
        subpop_idx: indices of h_vals where h_vals >= eta_opt.
    """
    m = len(h_vals)
    candidates = np.unique(h_vals)
    best_val = np.inf
    best_eta = None
    for eta in candidates:
        exceed = np.maximum(h_vals - eta, 0)
        val = eta + exceed.sum() / (alpha * m)
        if val < best_val:
            best_val = val
            best_eta = eta
    # define subpopulation as all points with h_vals >= best_eta
    subpop_idx = np.where(h_vals >= best_eta)[0]
    return best_val, best_eta, subpop_idx


def worst_case_group_loss(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    Z: Union[pd.DataFrame, np.ndarray],
    alpha: float = 0.05,
    cond_regressor=None,
    classification: bool = False,
    split_ratio: float = 0.5,
    random_state: Union[int, None] = None
) -> Tuple[float, float, List[int], List[int]]:
    """
    Two-sample estimator with role-swap averaging, returning subpopulations:

    Returns:
        W_avg: averaged worst-case subgroup loss,
        eta_avg: averaged threshold η,
        Q1: list of original indices for subpopulation from (S1→S2) split,
        Q2: list of original indices for subpopulation from (S2→S1) split.
    """
    n = len(y)
    rng = np.random.RandomState(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n1 = int(split_ratio * n)
    S1, S2 = idx[:n1], idx[n1:]

    def single_split(train_idx, eval_idx):
        # compute per-example losses on train_idx
        if classification:
            probs = model.predict_proba(X[train_idx])
            eps = 1e-12
            clipped = np.clip(probs, eps, 1 - eps)
            losses = -np.log(clipped[np.arange(len(train_idx)), y[train_idx]])
        else:
            preds = model.predict(X[train_idx])
            losses = (preds - y[train_idx]) ** 2
        # fit h and predict on eval
        h_mod = estimate_conditional_risk(losses, Z[train_idx], base_regressor=cond_regressor)
        h_vals = h_mod.predict(Z[eval_idx])
        W, eta, local_idx = compute_worst_cvar(h_vals, alpha)
        # map local positions to original indices
        subpop = list(eval_idx[local_idx])
        return W, eta, subpop

    W1, eta1, Q1 = single_split(S1, S2)
    W2, eta2, Q2 = single_split(S2, S1)

    W_avg = (W1 + W2) / 2
    eta_avg = (eta1 + eta2) / 2
    return W_avg, eta_avg, Q1, Q2