import numpy as np
from joblib import Parallel, delayed

def _build_var_design_multiY(X, p, add_const=True, demean=True):
    """
    Construct the design matrix for VAR(p) (multi-target regression version).
    Input
    X: (R, T) ROI x time for a single subject
    p: order of the VAR model
    add_const: whether to include a constant column
    demean: whether to remove the mean from each ROI first (a common practice)
    Return
    Y: (T_eff, R) targets (current time points, all ROIs)
    Z_full: (T_eff, K) predictors (concatenated lags; may include constant column)
    idx_cols_by_roi: list of length R, where the j-th element gives the column indices corresponding to the p lags of source ROI j (used for reduction)
    """
    R, T = X.shape
    if demean:
        X = X - X.mean(axis=1, keepdims=True)

    T_eff = T - p
    if T_eff <= 0:
        raise ValueError("T must be > p.")

    Y = X[:, p:].T  # (T_eff, R)

    # concat lagging block Z_lags = [X(t-1), X(t-2), ..., X(t-p)]
    blocks = []
    for lag in range(1, p + 1):
        blocks.append(X[:, p - lag:T - lag].T)  # (T_eff, R)
    Z_lags = np.hstack(blocks)  # (T_eff, p*R)

    idx_cols_by_roi = []
    for j in range(R):
        cols_j = [lag * R + j for lag in range(0, p)]  # t-1..t-p
        idx_cols_by_roi.append(cols_j)

    if add_const:
        Z_full = np.hstack([np.ones((T_eff, 1), dtype=Z_lags.dtype), Z_lags])  # (T_eff, 1+pR)
        idx_cols_by_roi = [[c + 1 for c in cols] for cols in idx_cols_by_roi]
    else:
        Z_full = Z_lags

    return Y, Z_full, idx_cols_by_roi


def _multi_target_ols_resid_var(Y, Z):
    """
    Multi-target least squares: fit all targets (columns) simultaneously and return the residual variance for each target.
    Input
    Y: (T_eff, R) targets
    Z: (T_eff, K) design matrix
    Return
    sigma2: (R,) unbiased residual variance for each target
    """
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    resid = Y - Z @ B                      # (T_eff, R)
    T_eff, K = Z.shape
    denom = max(T_eff - K, 1)
    sigma2 = np.einsum("ij,ij->j", resid, resid) / denom 
    return sigma2  # (R,)


def _gc_one_subject(X, order=1, add_const=True, demean=True):
    """
    Single-subject GC computation: return the matrix of GC_{j→i} values (R, R), with the diagonal set to 0.
    Implemented as: one multi-target regression for the full model + R multi-target regressions for the reduced models (each excluding source j).
    """
    R, T = X.shape

    p = int(order)
    while (T - p) <= (1 + p * R):
        p -= 1
        if p < 1:
            raise ValueError(
                f"Unidentifiable: with T={T}, R={R}, even p=1 violates T-p > 1+pR."
            )

    Y, Z_full, idx_cols_by_roi = _build_var_design_multiY(X, p=p, add_const=add_const, demean=demean)

    sigma2_full = _multi_target_ols_resid_var(Y, Z_full)   # (R,)

    gc_mat = np.zeros((R, R), dtype=np.float64)

    K_full = Z_full.shape[1]
    base_mask = np.ones(K_full, dtype=bool)

    for j in range(R):
        keep = base_mask.copy()
        keep[idx_cols_by_roi[j]] = False    
        Z_red = Z_full[:, keep]               # (T_eff, K_red)

        sigma2_red = _multi_target_ols_resid_var(Y, Z_red)  # (R,)

        # GC_{j->i} = log( sigma2_red(i) / sigma2_full(i) )
        ratio = np.maximum(sigma2_red, 1e-300) / np.maximum(sigma2_full, 1e-300)
        vals = np.log(ratio)
        vals[vals < 0] = 0.0
        gc_mat[:, j] = vals

    np.fill_diagonal(gc_mat, 0.0)
    return gc_mat


def compute_gc_fc_big(
    bold,              # (N, R, T)
    order=1,           
    add_const=True,
    demean=True,
    n_jobs=-1,         
    batch_size="auto"
):
    """
    Constructor for time-domain Granger-causal functional connectivity (conditional GC) for large fMRI datasets.

    Each subject performs only (1 + R) multi-target OLS fits (significantly reducing computation).

    Automatically checks identifiability and decreases the order if violated.

    Runs in parallel across subjects.
    Return
    gc_all: directed GC matrices (i<-j) with shape (N, R, R).    
    """
    if bold.ndim != 3:
        raise ValueError("bold should be (num_subjects, num_rois, num_timepoints).")

    N, R, T = bold.shape

    if (T - order) <= (1 + order * R):
        pass

    tasks = (bold[s].astype(np.float64, copy=False) for s in range(N))

    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=batch_size)(
        delayed(_gc_one_subject)(X, order=order, add_const=add_const, demean=demean)
        for X in tasks
    )

    gc_all = np.stack(results, axis=0)  # (N, R, R)
    return gc_all


# ----------------------------------------------
# Pairwise (bivariate) Granger causality (by ROI pair).
# Suitable for high-dimensional datasets with few time points, e.g., ABIDE/PNC.
# ----------------------------------------------
def compute_pairwise_gc_fc(
    bold,              # (N, R, T)
    order=1,
    add_const=True,
    demean=True,
    n_jobs=-1,      
    backend="loky",
    batch_size="auto",
):
    """
    Use a bivariate VAR(p) to compute, for each ROI pair (i, j), GC_{j->i} and GC_{i->j}, and fill them into an (R, R) directed matrix. Unlike conditional GC, this function does not condition on other ROIs, making it suitable for datasets with small T and large R (e.g., ABIDE/PNC).

    Return
    gc_all: (N, R, R)
    """
    if bold.ndim != 3:
        raise ValueError("bold should be (num_subjects, num_rois, num_timepoints).")

    N, R, T = bold.shape
    iu1, iu2 = np.triu_indices(R, k=1)

    gc_all = np.zeros((N, R, R), dtype=np.float64)

    for s in range(N):
        X = bold[s].astype(np.float64, copy=False)  # (R, T)

        def _pair_job(k):
            i = iu1[k]
            j = iu2[k]
            X2 = X[[i, j], :]                  # (2, T)
            gc2 = _gc_one_subject(
                X2, order=order, add_const=add_const, demean=demean
            )                                   # (2, 2)

            return i, j, float(gc2[0, 1]), float(gc2[1, 0])

        results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
            delayed(_pair_job)(k) for k in range(iu1.size)
        )

        G = np.zeros((R, R), dtype=np.float64)
        for i, j, gij, gji in results:
            G[i, j] = gij
            G[j, i] = gji

        np.fill_diagonal(G, 0.0)

        gc_all[s] = G

    return gc_all
