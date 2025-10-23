import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from scipy.signal import welch, get_window

from load_various_data import load_abide_data, load_abcd_data, load_pnc_data, load_hcp_data



def _ensure_output_dir(base_dir: Path, dataset: str, method: str) -> Path:
    out_dir = base_dir / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _clamp_range(start_idx: int, end_idx_inclusive: int, num_subjects: int) -> Tuple[int, int]:
    if num_subjects <= 0:
        raise ValueError("No subjects found in input BOLD.")
    s = max(0, int(start_idx))
    e = min(int(end_idx_inclusive), num_subjects - 1)
    if s > e:
        raise ValueError(f"Invalid index range after clamping: start={s}, end={e}, num_subjects={num_subjects}")
    return s, e


def _save_node_feature(out_dir: Path, dataset: str, method: str, subject_index: int, node_feat: np.ndarray, overwrite: bool) -> Path:
    out_path = out_dir / f"{dataset}_{method}_subj_{subject_index:06d}.npy"
    if out_path.exists() and not overwrite:
        return out_path
    np.save(out_path, node_feat)
    return out_path


def _load_norm_stats(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    mean = data['mean']
    std = data['std']
    return mean, std


def _save_norm_stats(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)


def _apply_zscore(feat: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # feat: (R, D)
    return (feat - mean[None, :]) / (std[None, :] + eps)


# ----------------------------
# stat feature (temporal domain)
# ----------------------------

def _compute_stat_features_single_roi(x: np.ndarray) -> np.ndarray:
    """
    Input: x, shape (T,)
    Output: features, shape (D_stat=16,)
    first 8 dim: mean, std, cv, skewness, kurtosis, acf1, acf2, ar1
    second 8 dim: median, p25, p75, iqr, range, zcr, hjorth_mob, hjorth_comp
    """
    if x.ndim != 1:
        raise ValueError("ROI signal must be 1D time series.")

    T = x.shape[0]
    eps = 1e-8

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    cv = float(std / (abs(mean) + eps))

    if std < eps:
        skew = 0.0
        kurt = 0.0
    else:
        z = (x - mean) / (std + eps)
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)  # excess kurtosis

    def acf_at_lag(sig: np.ndarray, lag: int) -> float:
        if lag <= 0 or lag >= sig.size:
            return 0.0
        xm = sig - sig.mean()
        denom = float(np.dot(xm, xm) + eps)
        num = float(np.dot(xm[:-lag], xm[lag:]))
        return num / denom

    acf1 = acf_at_lag(x, 1)
    acf2 = acf_at_lag(x, 2)

    # AR(1) coeff
    if T >= 3:
        x_t = x[1:]
        x_tm1 = x[:-1]
        denom = float(np.dot(x_tm1, x_tm1) + eps)
        ar1 = float(np.dot(x_tm1, x_t) / denom)
    else:
        ar1 = 0.0

    median = float(np.median(x))
    p25 = float(np.percentile(x, 25))
    p75 = float(np.percentile(x, 75))
    iqr = float(p75 - p25)
    x_range = float(np.max(x) - np.min(x))

    if T >= 2:
        signs = np.sign(x)
        for i in range(1, T):
            if signs[i] == 0:
                signs[i] = signs[i-1]
        zcr = float(np.sum(signs[1:] * signs[:-1] < 0) / (T - 1))
    else:
        zcr = 0.0

    # Hjorth
    if T >= 3:
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x = float(np.var(x))
        var_dx = float(np.var(dx))
        var_ddx = float(np.var(ddx))
        hj_mob = float(np.sqrt(var_dx / (var_x + eps)))
        hj_comp = float(np.sqrt(var_ddx / (var_dx + eps)) / (hj_mob + eps))
    else:
        hj_mob = 0.0
        hj_comp = 0.0

    return np.array([
        mean, std, cv, skew, kurt, acf1, acf2, ar1,
        median, p25, p75, iqr, x_range, zcr, hj_mob, hj_comp
    ], dtype=np.float32)


def compute_statistical_features(bold: np.ndarray) -> np.ndarray:
    """
    Input: bold, shape (N, R, T)
    Output: feats, shape (N, R, D_stat=16)
    """
    if bold.ndim != 3:
        raise ValueError("Expected bold with shape (num_subjects, num_rois, num_timepoints).")

    N, R, T = bold.shape
    D = 16
    out = np.zeros((N, R, D), dtype=np.float32)

    for s in range(N):
        X = bold[s]  # (R, T)
        for r in range(R):
            out[s, r] = _compute_stat_features_single_roi(X[r])
    return out


# ----------------------------
# Welch feature
# ----------------------------

def _pick_welch_params(T: int, nperseg: Optional[int], noverlap: Optional[int]) -> Tuple[int, int]:
    if nperseg is None:
        target_segments = 6
        overlap = 0.5
        nperseg = max(8, int(T / (target_segments * (1 - overlap))))
        nperseg = min(nperseg, max(8, T // 2))
    if noverlap is None:
        noverlap = max(0, int(0.5 * nperseg))
    nperseg = max(4, min(nperseg, T))
    noverlap = max(0, min(noverlap, nperseg - 1))
    return nperseg, noverlap


def _band_power(f: np.ndarray, Pxx: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    df = np.mean(np.diff(f)) if f.size > 1 else 1.0
    return float(np.sum(Pxx[mask]) * df)


def _spectral_entropy(Pxx: np.ndarray, eps: float = 1e-12) -> float:
    p = Pxx.astype(np.float64)
    p = p / (p.sum() + eps)
    return float(-np.sum(p * np.log(p + eps)))


def _spectral_centroid(f: np.ndarray, Pxx: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.sum(Pxx) + eps)
    return float(np.sum(f * Pxx) / denom)


def _spectral_bandwidth(f: np.ndarray, Pxx: np.ndarray, centroid: Optional[float] = None, eps: float = 1e-12) -> float:
    if centroid is None:
        centroid = _spectral_centroid(f, Pxx, eps=eps)
    return float(np.sqrt(np.sum(((f - centroid) ** 2) * Pxx) / (np.sum(Pxx) + eps)))


def _one_over_f_slope(f: np.ndarray, Pxx: np.ndarray) -> float:
    mask = (f > 0) & np.isfinite(Pxx) & (Pxx > 0)
    if np.sum(mask) < 3:
        return 0.0
    xf = np.log(f[mask])
    yf = np.log(Pxx[mask])
    A = np.vstack([xf, np.ones_like(xf)]).T
    slope, _ = np.linalg.lstsq(A, yf, rcond=None)[0]
    return float(slope)


def _spectral_flatness(Pxx: np.ndarray, eps: float = 1e-12) -> float:
    gm = float(np.exp(np.mean(np.log(Pxx + eps))))
    am = float(np.mean(Pxx + eps))
    return float(gm / am)


def _rolloff_frequency(f: np.ndarray, Pxx: np.ndarray, percentile: float = 0.85) -> float:
    if f.size == 0:
        return 0.0
    df = np.mean(np.diff(f)) if f.size > 1 else 1.0
    cum = np.cumsum(Pxx) * df
    total = cum[-1] if cum.size > 0 else 1.0
    target = percentile * (total if total > 0 else 1.0)
    idx = np.searchsorted(cum, target, side='left')
    idx = int(min(max(idx, 0), f.size - 1))
    return float(f[idx])


def _compute_psd_features_single_roi(x: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: Optional[int], noverlap: Optional[int], window: str = "hann", detrend: str = "constant") -> np.ndarray:
    """
    PSD feature dim=16:
    First 10 dimensions: ALFF, fALFF, Slow-5/4/3 power, spectral entropy, spectral centroid, bandwidth, 1/f slope, peak frequency
    Last 6 dimensions: relative Slow-5/4/3 power, spectral flatness, 85% roll-off frequency, peak power
    """
    T = x.shape[0]
    nperseg_eff, noverlap_eff = _pick_welch_params(T, nperseg, noverlap)

    f, Pxx = welch(x, fs=fs, window=get_window(window, nperseg_eff), nperseg=nperseg_eff, noverlap=noverlap_eff, detrend=detrend, axis=-1)

    nyq = 0.5 * fs
    def clamp_band(lo: float, hi: float) -> Tuple[float, float]:
        lo_c = max(0.0, lo)
        hi_c = min(nyq, hi)
        if hi_c <= lo_c:
            return lo_c, lo_c - 1e-6
        return lo_c, hi_c

    band_alff = clamp_band(0.01, 0.08)
    band_s5   = clamp_band(0.01, 0.027)
    band_s4   = clamp_band(0.027, 0.073)
    band_s3   = clamp_band(0.073, 0.198)

    total_power = _band_power(f, Pxx, 0.0, nyq)
    alff = _band_power(f, Pxx, *band_alff)
    falff = float(alff / (total_power + 1e-12))

    p_s5 = _band_power(f, Pxx, *band_s5)
    p_s4 = _band_power(f, Pxx, *band_s4)
    p_s3 = _band_power(f, Pxx, *band_s3)

    rp_s5 = float(p_s5 / (total_power + 1e-12))
    rp_s4 = float(p_s4 / (total_power + 1e-12))
    rp_s3 = float(p_s3 / (total_power + 1e-12))

    sent = _spectral_entropy(Pxx)
    sc = _spectral_centroid(f, Pxx)
    sbw = _spectral_bandwidth(f, Pxx, centroid=sc)
    slope = _one_over_f_slope(f, Pxx)

    if Pxx.size > 0:
        peak_idx = int(np.argmax(Pxx))
        peak_freq = float(f[peak_idx])
        peak_power = float(Pxx[peak_idx])
    else:
        peak_freq = 0.0
        peak_power = 0.0

    flatness = _spectral_flatness(Pxx)
    rolloff85 = _rolloff_frequency(f, Pxx, percentile=0.85)

    return np.array([
        alff, falff, p_s5, p_s4, p_s3, sent, sc, sbw, slope, peak_freq,
        rp_s5, rp_s4, rp_s3, flatness, rolloff85, peak_power
    ], dtype=np.float32)


def compute_psd_features(bold: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: Optional[int], noverlap: Optional[int], window: str, detrend: str) -> np.ndarray:
    if bold.ndim != 3:
        raise ValueError("Expected bold with shape (num_subjects, num_rois, num_timepoints).")
    N, R, T = bold.shape
    D = 16
    out = np.zeros((N, R, D), dtype=np.float32)
    for s in range(N):
        X = bold[s]
        for r in range(R):
            out[s, r] = _compute_psd_features_single_roi(
                X[r], fs=fs, fmin=fmin, fmax=fmax, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend
            )
    return out


def load_bold_data(dataset: str) -> Tuple[np.ndarray, Optional[float]]:
    
    if dataset == "ABIDE":
        bold = load_abide_data()
        fs = 1.0 / 2.0
        return bold, fs

    if dataset == "HCP":
        bold = load_hcp_data()
        fs = 1.0 / 0.72

        return bold, fs
    
    if dataset == "PNC":
        bold = load_pnc_data()
        fs = 1.0 / 3.0
        return bold, fs
    
    if dataset == "ABCD":
        bold = load_abcd_data()
        fs = 1.0 / 0.8
        return bold, fs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct per-ROI node features (statistical or PSD) for subjects in an index range and save per-subject outputs.\n"
            "Input BOLD shape: (num_subjects, num_rois, num_timepoints).\n"
            "Outputs saved under {output_dir}/{dataset}/{method}/ as Numpy .npy files."
        )
    )

    parser.add_argument("--dataset", required=True, choices=["ABCD", "ABIDE", "PNC", "HCP", "CUSTOM"], help="Dataset name to load.")
    parser.add_argument("--method", required=True, choices=["stat", "psd"], help="Node feature construction method: 'stat' or 'psd'.")
    parser.add_argument("--start", type=int, required=True, help="Start subject index (inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End subject index (inclusive).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save per-subject node features.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-subject outputs.")

    parser.add_argument("--input_npy", type=str, default=None, help="Optional: path to an .npy containing (N,R,T) BOLD. Overrides dataset loader if provided.")

    parser.add_argument("--fs", type=float, default=None, help="Sampling rate (Hz). If not given, provide --tr or let loader return fs.")
    parser.add_argument("--tr", type=float, default=None, help="Repetition time (sec). If given, fs=1/TR.")

    # PSD parameter
    parser.add_argument("--fmin", type=float, default=0.01, help="Lower frequency bound for band definitions (Hz).")
    parser.add_argument("--fmax", type=float, default=0.10, help="Upper frequency bound (unused in current features; kept for parity).")
    parser.add_argument("--nperseg", type=int, default=None, help="Welch nperseg; default auto.")
    parser.add_argument("--noverlap", type=int, default=None, help="Welch noverlap; default 50% of nperseg.")
    parser.add_argument("--detrend", type=str, default="constant", choices=["constant", "linear", "none"], help="Welch detrend mode.")
    parser.add_argument("--window", type=str, default="hann", help="Welch window name.")

    # normalization
    parser.add_argument("--normalize", action="store_true", help="Apply dataset-level z-score per feature using stats over selected subjects and all ROIs.")
    parser.add_argument("--refit_norm", action="store_true", help="Recompute and overwrite normalization stats even if existing.")
    parser.add_argument("--norm_stats_path", type=str, default=None, help="Path to save/load normalization stats .npz. Default: {output_dir}/{dataset}/{method}/_norm_stats.npz")

    return parser.parse_args()


def _resolve_fs(fs_arg: Optional[float], tr_arg: Optional[float], fs_from_loader: Optional[float], needs_fs: bool) -> Optional[float]:
    if fs_arg is not None:
        return float(fs_arg)
    if tr_arg is not None:
        if tr_arg <= 0:
            raise ValueError("--tr must be > 0")
        return 1.0 / float(tr_arg)
    if needs_fs and fs_from_loader is None:
        raise ValueError("fs is required for PSD features. Provide --fs or --tr, or return fs from loader.")
    return fs_from_loader


def _compute_features_for_subject_slice(method: str, bold_slice: np.ndarray, fs: Optional[float], args: argparse.Namespace) -> np.ndarray:
    # bold_slice: (1, R, T) -> returns (R, D)
    if method == "stat":
        return compute_statistical_features(bold_slice)[0]
    if method == "psd":
        if fs is None:
            raise ValueError("PSD features require fs.")
        return compute_psd_features(
            bold_slice, fs=fs, fmin=args.fmin, fmax=args.fmax, nperseg=args.nperseg, noverlap=args.noverlap, window=args.window, detrend=args.detrend
        )[0]
    raise ValueError(f"Unknown method: {method}")



def main():
    args = parse_args()

    # load data
    if args.input_npy is not None:
        bold = np.load(args.input_npy)
        fs_from_loader = None
        if bold.ndim != 3:
            raise ValueError("--input_npy must contain an array of shape (num_subjects, num_rois, num_timepoints)")
    else:
        bold, fs_from_loader = load_bold_data(args.dataset)

    N, R, T = bold.shape

    needs_fs = (args.method == "psd")
    fs = _resolve_fs(args.fs, args.tr, fs_from_loader, needs_fs)

    start_idx, end_idx = _clamp_range(args.start, args.end, N)

    out_dir = _ensure_output_dir(Path(args.output_dir), args.dataset, args.method)

    norm_stats_path = Path(args.norm_stats_path) if args.norm_stats_path else (out_dir / "_norm_stats.npz")

    print(f"Total subjects: {N}; processing indices [{start_idx}..{end_idx}] inclusive.")
    print(f"Dataset: {args.dataset}; Method: {args.method}; Output dir: {out_dir}")

    mean_vec = None
    std_vec = None
    if args.normalize:
        if norm_stats_path.exists() and not args.refit_norm:
            mean_vec, std_vec = _load_norm_stats(norm_stats_path)
            print(f"Loaded normalization stats from {norm_stats_path} with shape {mean_vec.shape}.")
        else:
            print("Fitting normalization stats over selected subjects and all ROIs...")
            feats_list = []
            for s in range(start_idx, end_idx + 1):
                subj_feat = _compute_features_for_subject_slice(args.method, bold[s:s+1], fs, args)  # (R, D)
                feats_list.append(subj_feat)
            all_feats = np.concatenate(feats_list, axis=0)  # (num_subj*R, D)
            mean_vec = all_feats.mean(axis=0)
            std_vec = all_feats.std(axis=0, ddof=0)
            _save_norm_stats(norm_stats_path, mean_vec, std_vec)
            print(f"Saved normalization stats to {norm_stats_path}.")

    num_done = 0
    num_skipped = 0
    for s in range(start_idx, end_idx + 1):
        out_path = out_dir / f"{args.dataset}_{args.method}_subj_{s:06d}.npy"
        if out_path.exists() and not args.overwrite:
            num_skipped += 1
            continue

        subj_feat = _compute_features_for_subject_slice(args.method, bold[s:s+1], fs, args)  # (R, D)

        if args.normalize:
            if mean_vec is None or std_vec is None:
                raise RuntimeError("Normalization requested but stats are missing.")
            subj_feat = _apply_zscore(subj_feat, mean_vec, std_vec)

        _save_node_feature(out_dir, args.dataset, args.method, s, subj_feat, overwrite=args.overwrite)
        num_done += 1
        print(f"Saved: {out_path}  (done={num_done}, skipped={num_skipped})")

    print(f"Completed. Newly processed: {num_done}, skipped existing: {num_skipped}.")


if __name__ == "__main__":
    main()
