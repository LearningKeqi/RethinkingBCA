import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def bandpass_filter(data, fs, fmin, fmax, order=4):
    """Zero-phase Butterworth band-pass filtering (applied independently to each time series)."""
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    if not (0 < low < high < 1):
        raise ValueError("fmin/fmax must satisfy 0 < fmin < fmax < fs/2.")
    b, a = butter(order, [low, high], btype="bandpass")

    return filtfilt(b, a, data, axis=-1, method="gust")

def compute_plv_fc(
    bold: np.ndarray,
    fs: float,
    fmin: float = 0.01,
    fmax: float = 0.10,
    filter_order: int = 4,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Construct functional connectivity (symmetric, in [0,1]) per subject using Phase Locking Value (PLV) within the band [fmin, fmax].

    Parameters
    bold: array, shape (N, R, T)
    N subjects, R ROIs, T time points.
    fs: float
    Sampling rate (Hz) = 1 / TR.
    fmin, fmax: float
    Target band (Hz), commonly 0.01–0.1.
    filter_order: int
    Butterworth filter order.
    eps: float
    Numerical safeguard to avoid zero amplitude in extreme cases.

    Returns
    plv_fc: array, shape (N, R, R)
    One PLV FC matrix per subject (symmetric, main diagonal = 1, values ∈ [0,1]).
    """
    if bold.ndim != 3:
        raise ValueError("bold should be (num_subjects, num_rois, num_timepoints).")
    N, R, T = bold.shape

    plv_fc = np.empty((N, R, R), dtype=np.float64)

    for s in range(N):
        X = bold[s]  # (R, T)

        if np.isnan(X).any():
            X = X.copy()
            m = np.nanmean(X, axis=1, keepdims=True)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take_along_axis(m, np.where(nan_mask)[1][None, :], axis=1)

        # 1) band pass
        X_bp = bandpass_filter(X, fs=fs, fmin=fmin, fmax=fmax, order=filter_order)  # (R, T)

        # 2) Hilbert trans
        analytic = hilbert(X_bp, axis=-1)  # (R, T) complex
        # z = e^{i phi}
        amp = np.abs(analytic)
        z = analytic / (amp + eps)  # (R, T) complex

        # PLV = | (1/T) * z @ z^H |
        ZZT = (z @ np.conj(z.T)) / T  # (R, R) complex
        P = np.abs(ZZT).astype(np.float64)  # (R, R) in [0,1]
        np.fill_diagonal(P, 1.0)          

        plv_fc[s] = P

    return plv_fc