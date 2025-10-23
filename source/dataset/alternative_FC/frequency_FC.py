import numpy as np
from scipy.signal import coherence, get_window
from joblib import Parallel, delayed

def _pair_coh(xi, xj, fs, window, nperseg, noverlap, detrend, band_idx):
    f, Cxy = coherence(
        xi, xj,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=-1,
    )
    val = float(np.nanmean(Cxy[band_idx])) if np.any(band_idx) else float(np.nanmean(Cxy))
    return val



def _choose_welch_params(T, target_segments=6, min_segments=3, overlap=0.5):
    """
    Given the number of time points T, adaptively choose nperseg and noverlap so that the
    number of Welch segments is ≥ min_segments. Use 50% overlap by default (configurable).
    """
    nperseg = max(8, int(T / (target_segments * (1 - overlap)))) 
    nperseg = min(nperseg, T // 2) if T >= 16 else max(4, T // 2)
    noverlap = int(nperseg * overlap)

    def n_segments(T, nperseg, noverlap):
        step = max(1, nperseg - noverlap)
        if nperseg > T: return 1
        return 1 + (T - nperseg) // step

    while n_segments(T, nperseg, noverlap) < min_segments and nperseg > 8:
        nperseg = max(8, nperseg - 1)
        noverlap = int(nperseg * overlap)

    return nperseg, noverlap



def compute_coherence_fc_fast(
    bold: np.ndarray,
    fs: float,
    fmin: float = 0.01,
    fmax: float = 0.10,
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str = "constant",
    window: str = "hann",
    n_jobs: int = -1,
    backend: str = "loky",
    batch_size: str | int = "auto",
) -> np.ndarray:
    """
    Compute the FC matrix based on magnitude-squared coherence in parallel (band-averaged).

    Parameters
    ----------
    bold : (N, R, T)
        N subjects, R ROIs, T time points.
    fs : float
        Sampling rate in Hz (fs = 1 / TR).
    fmin, fmax : float
        Frequency band (Hz) over which to average (e.g., 0.01–0.10).
    nperseg, noverlap, detrend, window
        Parameters passed through to `scipy.signal.coherence`; keep consistent with the original implementation.
    n_jobs : int
        Number of parallel processes/threads; -1 uses as many cores as possible.
    backend : str
        joblib backend; "loky" (multiprocessing, better isolation) or "threading" (may be limited by the GIL).
    batch_size : {"auto", int}
        Task batch size for joblib submission.

    Returns
    -------
    fc : (N, R, R)
    For each subject, an ROI×ROI coherence matrix averaged over the band; values in [0, 1].

    """
    if bold.ndim != 3:
        raise ValueError("Expected bold with shape (num_subjects, num_rois, num_timepoints).")

    N, R, T = bold.shape

    # nperseg, noverlap = _choose_welch_params(T)

    # print(f'nperseg={nperseg}, noverlap={noverlap}')

    win = window

    test_sig_1 = bold[0, 0]
    test_sig_2 = bold[0, min(1, R-1)]
    f_probe, _ = coherence(
        test_sig_1, test_sig_2,
        fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, detrend=detrend, axis=-1
    )
    band_idx = (f_probe >= fmin) & (f_probe <= fmax)

    iu1, iu2 = np.triu_indices(R, k=1)

    fc = np.zeros((N, R, R), dtype=float)

    for s in range(N):
        X = bold[s]  # (R, T)

        vals = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
            delayed(_pair_coh)(
                X[iu1[k]], X[iu2[k]],
                fs, win, nperseg, noverlap, detrend, band_idx
            )
            for k in range(iu1.size)
        )

        subj_fc = np.eye(R, dtype=float)
        subj_fc[iu1, iu2] = vals
        subj_fc[iu2, iu1] = vals

        fc[s] = subj_fc

    return fc

