import numpy as np
from joblib import Parallel, delayed

def _fit_mvar_statsmodels(X, order):
    """
    Use statsmodels to fit an MVAR model: input X with shape (R, T) → return A_coefs with shape (p, R, R) and noise_cov with shape (R, R).
    """
    try:
        from statsmodels.tsa.api import VAR
    except Exception as e:
        raise ImportError("This function requires 'statsmodels'. Install via `pip install statsmodels`.") from e

    Xz = X.T - X.T.mean(axis=0, keepdims=True) 
    model = VAR(Xz)
    res = model.fit(maxlags=order, ic=None, trend='n')
    # res.coefs: (p, R, R) ; res.sigma_u: (R, R)
    A = res.coefs.copy()
    Sigma = res.sigma_u.copy()
    return A, Sigma


def _A_of_f(A_coefs, freqs, fs):
    """
    Calculate A(f) = I - sum_k A_k * exp(-i*2π f k / fs)
    A_coefs: (p, R, R)
    freqs: (F,) Hz
    return Af: (F, R, R)
    """
    p, R, _ = A_coefs.shape
    F = len(freqs)
    Af = np.zeros((F, R, R), dtype=np.complex128)
    I = np.eye(R, dtype=np.complex128)
    for fi, f in enumerate(freqs):
        w = 2.0 * np.pi * f / fs
        Af_f = I.copy()
        for k in range(1, p + 1):
            Af_f -= A_coefs[k - 1] * np.exp(-1j * w * k)
        Af[fi] = Af_f
    return Af


def _pdc_from_Af(Af, eps=1e-15, use_gpdc=False, noise_cov=None):
    """
    from A(f) to obtian PDC
    Af: (F, R, R) 
    return PDC: (F, R, R),i<-j
    """
    F, R, _ = Af.shape
    P = np.empty_like(Af.real)  # (F, R, R)

    if use_gpdc:
        if noise_cov is None:
            raise ValueError("use_gpdc=True need noise_cov (R, R)")
        # pre-calculate Sigma^{-1}
        from numpy.linalg import inv
        Sigma_inv = inv(noise_cov + 1e-15 * np.eye(noise_cov.shape[0]))
    else:
        Sigma_inv = None

    for fi in range(F):
        A = Af[fi]  # (R, R)
        if use_gpdc:
            for j in range(R):
                col = A[:, j]  # (R,)
                den = np.sqrt(np.real(np.conj(col).T @ (Sigma_inv @ col)) + eps)
                P[fi, :, j] = np.abs(col) / den
        else:
            den = np.sqrt(np.sum(np.abs(A)**2, axis=0) + eps)  # (R,)
            P[fi] = np.abs(A) / den[None, :]
    return P  # (F, R, R)


def compute_pdc_fc(
    bold: np.ndarray,
    fs: float,
    order: int = 5,
    fmin: float = 0.01,
    fmax: float = 0.10,
    n_freqs: int = 256,
    use_gpdc: bool = False,
) -> np.ndarray:
    """
    Compute directed FC per subject using Partial Directed Coherence (PDC) averaged over a given frequency band.

    Parameters
    bold: array, shape (N, R, T)
    BOLD signals with N subjects, R ROIs, and T time points.
    fs: float
    Sampling rate (Hz) = 1 / TR.
    order: int
    MVAR order p. Can be chosen by experience/information criteria (AIC/BIC); here a fixed value is used by default.
    fmin, fmax: float
    Frequency band (Hz) over which PDC is averaged (e.g., 0.01–0.10 Hz).
    n_freqs: int
    Number of frequency samples (uniformly spaced over [0, fs/2]).
    use_gpdc: bool
    Whether to use gPDC (with noise-covariance normalization). False uses standard PDC.

    Returns
    fc_dir: array, shape (N, R, R)
    Directed FC (i←j) matrix for each subject; diagonal set to 0.
    """
    if bold.ndim != 3:
        raise ValueError("Expected bold with shape (num_subjects, num_rois, num_timepoints).")

    N, R, T = bold.shape

    freqs = np.linspace(0.0, fs / 2.0, n_freqs, endpoint=True)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        raise ValueError("Frequency grid does not cover the requested band; adjust n_freqs/fmin/fmax.")

    fc_dir = np.zeros((N, R, R), dtype=float)

    for s in range(N):
        X = bold[s]  # (R, T)
        # fit MVAR
        A_coefs, Sigma = _fit_mvar_statsmodels(X, order=order)  # (p,R,R), (R,R)

        # calculate A(f) and PDC(f)
        Af = _A_of_f(A_coefs, freqs=freqs, fs=fs)              # (F,R,R) complex
        P = _pdc_from_Af(Af, use_gpdc=use_gpdc, noise_cov=Sigma)  # (F,R,R) real

        # band average
        P_band = P[band_mask].mean(axis=0)  # (R,R)
        np.fill_diagonal(P_band, 0.0)      
        fc_dir[s] = P_band

    return fc_dir


def compute_pairwise_pdc_fc(
    bold: np.ndarray,
    fs: float,
    order: int = 1,
    fmin: float = 0.01,
    fmax: float = 0.10,
    n_freqs: int = 256,
    use_gpdc: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
    batch_size: str | int = "auto",
) -> np.ndarray:
    """
    Pairwise (bivariate) PDC: for each ROI pair (i, j), fit a 2-variable MVAR(p), compute the band-averaged PDC, and fill the result into the global (R, R) directed matrix (i←j).

    Return
    fc_dir: (N, R, R)
    """
    if bold.ndim != 3:
        raise ValueError("Expected bold with shape (num_subjects, num_rois, num_timepoints).")

    N, R, T = bold.shape

    freqs = np.linspace(0.0, fs / 2.0, n_freqs, endpoint=True)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        raise ValueError("Frequency grid does not cover the requested band; adjust n_freqs/fmin/fmax.")

    iu1, iu2 = np.triu_indices(R, k=1)
    fc_all = np.zeros((N, R, R), dtype=float)

    for s in range(N):
        X = bold[s]  # (R, T)

        def _pair_job(k: int):
            i = iu1[k]
            j = iu2[k]
            X2 = X[[i, j], :]  # (2, T)
            A_coefs, Sigma = _fit_mvar_statsmodels(X2, order=order)
            Af = _A_of_f(A_coefs, freqs=freqs, fs=fs)
            P = _pdc_from_Af(Af, use_gpdc=use_gpdc, noise_cov=Sigma)  # (F, 2, 2)
            P_band = P[band_mask].mean(axis=0)  # (2, 2)
            P_band[0, 0] = 0.0
            P_band[1, 1] = 0.0
            return i, j, float(P_band[0, 1]), float(P_band[1, 0])

        results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
            delayed(_pair_job)(k) for k in range(iu1.size)
        )

        G = np.zeros((R, R), dtype=float)
        for i, j, gij, gji in results:
            G[i, j] = gij
            G[j, i] = gji
        np.fill_diagonal(G, 0.0)

        fc_all[s] = G

    return fc_all
