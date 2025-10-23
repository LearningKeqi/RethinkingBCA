import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from frequency_FC import compute_coherence_fc_fast
from granger_causality_FC import compute_gc_fc_big, compute_pairwise_gc_fc
from partial_directed_coherence_FC import compute_pdc_fc, compute_pairwise_pdc_fc
from phase_based_FC import compute_plv_fc
from test_new_gc import granger_causality_fstat_fast_maxF_minP_parallel

from load_various_data import load_abide_data, load_hcp_data, load_pnc_data, load_abcd_data


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


def _ensure_output_dir(base_dir: Path, dataset: str, method: str) -> Path:
    out_dir = base_dir / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_fs(fs_arg: Optional[float], tr_arg: Optional[float], fs_from_loader: Optional[float], method: str) -> Optional[float]:
    """
    Obtain fs through parsing. For methods that require frequency information (frequency, phase, pdc), fs must be obtained.
    """
    if fs_arg is not None:
        return float(fs_arg)
    if tr_arg is not None:
        if tr_arg <= 0:
            raise ValueError("--tr must be > 0")
        return 1.0 / float(tr_arg)
    if method in {"frequency", "phase", "pdc"}:
        if fs_from_loader is None:
            raise ValueError("fs is required for method '%s'. Provide --fs or --tr, or return fs from loader." % method)
        return float(fs_from_loader)
    return fs_from_loader


def _clamp_range(start_idx: int, end_idx_inclusive: int, num_subjects: int) -> Tuple[int, int]:
    if num_subjects <= 0:
        raise ValueError("No subjects found in input BOLD.")
    s = max(0, int(start_idx))
    e = min(int(end_idx_inclusive), num_subjects - 1)
    if s > e:
        raise ValueError(f"Invalid index range after clamping: start={s}, end={e}, num_subjects={num_subjects}")
    return s, e


def _save_fc_matrix(out_dir: Path, dataset: str, method: str, subject_index: int, fc: np.ndarray, overwrite: bool) -> Path:
    out_path = out_dir / f"{dataset}_{method}_subj_{subject_index:06d}.npy"
    if out_path.exists() and not overwrite:
        return out_path
    np.save(out_path, fc)
    return out_path


def _compute_single_subject_fc(method: str, X: np.ndarray, fs: Optional[float], args: argparse.Namespace, dataset: Optional[str] = None) -> np.ndarray:
    """
    Compute the FC for a single subject, returning a matrix of shape (R, R).
    X: input data with shape (R, T).    """
    if X.ndim != 2:
        raise ValueError("Single subject data X must have shape (num_rois, num_time_points).")

    if method == "frequency":
        if fs is None:
            raise ValueError("frequency method requires fs")
        fc = compute_coherence_fc_fast(
            X[None, :, :],
            fs=fs,
            fmin=args.fmin,
            fmax=args.fmax,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            detrend=args.detrend,
            window=args.window,
            n_jobs=args.inner_jobs,
            backend=args.backend,
            batch_size="auto",
        )[0]
        return fc

    if method == "phase":
        if fs is None:
            raise ValueError("phase method requires fs")
        fc = compute_plv_fc(
            X[None, :, :],
            fs=fs,
            fmin=args.fmin,
            fmax=args.fmax,
            filter_order=args.filter_order,
        )[0]
        return fc

    if method == "gc":
        use_pairwise = (dataset in {"ABIDE", "PNC"}) if dataset is not None else False
        if use_pairwise:
            fc = compute_pairwise_gc_fc(
                X[None, :, :],
                order=args.order,
                add_const=not args.no_const,
                demean=not args.no_demean,
                n_jobs=-1,
                backend="loky",
            )[0]
        else:
            fc = compute_gc_fc_big(
                X[None, :, :],
                order=args.order,
                add_const=not args.no_const,
                demean=not args.no_demean,
                n_jobs=1,
            )[0]
        return fc

    if method == "pdc":
        if fs is None:
            raise ValueError("pdc method requires fs")
        use_pairwise = (dataset in {"ABIDE", "PNC"}) if dataset is not None else False
        if use_pairwise:
            fc = compute_pairwise_pdc_fc(
                X[None, :, :],
                fs=fs,
                order=args.order,
                fmin=args.fmin,
                fmax=args.fmax,
                n_freqs=args.n_freqs,
                use_gpdc=args.use_gpdc,
                n_jobs=-1,
                backend="loky",
            )[0]
        else:
            fc = compute_pdc_fc(
                X[None, :, :],
                fs=fs,
                order=args.order,
                fmin=args.fmin,
                fmax=args.fmax,
                n_freqs=args.n_freqs,
                use_gpdc=args.use_gpdc,
            )[0]
        return fc

    if method == "gc_fromSL":
        F_mat, P_mat = granger_causality_fstat_fast_maxF_minP_parallel(X, max_lag=2)
        return F_mat, P_mat

    raise ValueError(f"Unknown method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct Functional Connectivity (FC) for a subject index range and save per-subject outputs.\n"
            "Existing methods: frequency (coherence), phase (PLV), gc (Granger causality), pdc (Partial Directed Coherence).\n"
            "Index range is inclusive: e.g., --start 0 --end 100 processes subjects [0..100]."
        )
    )

    parser.add_argument("--dataset", required=True, choices=["ABCD", "ABIDE", "PNC", "HCP", "CUSTOM"], help="Dataset name to load. Implement loader accordingly.")
    parser.add_argument("--method", required=True, choices=["frequency", "phase", "gc", "pdc", "gc_fromSL"], help="Which FC method to use.")
    parser.add_argument("--start", type=int, required=True, help="Start subject index (inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End subject index (inclusive).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save per-subject FC matrices.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-subject outputs.")

    parser.add_argument("--input_npy", type=str, default=None, help="Optional: path to an .npy containing (N,R,T) BOLD. Overrides dataset loader if provided.")

    parser.add_argument("--fs", type=float, default=None, help="Sampling rate (Hz). If not given, provide --tr or let loader return fs.")
    parser.add_argument("--tr", type=float, default=None, help="Repetition time (sec). If given, fs=1/TR.")

    # frequency/phase/pdc
    parser.add_argument("--fmin", type=float, default=0.01, help="Lower frequency bound (Hz).")
    parser.add_argument("--fmax", type=float, default=0.10, help="Upper frequency bound (Hz).")

    # frequency-specific param
    parser.add_argument("--nperseg", type=int, default=None, help="coherence nperseg; default=min(T,256).")
    parser.add_argument("--noverlap", type=int, default=None, help="coherence noverlap; default=nperseg//2.")
    parser.add_argument("--detrend", type=str, default="constant", choices=["constant", "linear", "none"], help="coherence detrend mode.")
    parser.add_argument("--window", type=str, default="hann", help="coherence window name.")
    parser.add_argument("--inner_jobs", type=int, default=-1, help="Inner parallelism for ROI-pair computations where applicable.")
    parser.add_argument("--backend", type=str, default="loky", choices=["loky", "threading"], help="Joblib backend for inner parallelism.")

    # phase-specific param
    parser.add_argument("--filter_order", type=int, default=4, help="Butterworth bandpass filter order for PLV.")

    # gc-specific param
    parser.add_argument("--order", type=int, default=1, help="VAR/MVAR model order (method-dependent).")
    parser.add_argument("--no_const", action="store_true", help="GC: do not include constant column.")
    parser.add_argument("--no_demean", action="store_true", help="GC: do not demean time series.")

    # pdc-specific param
    parser.add_argument("--n_freqs", type=int, default=256, help="Number of frequency samples for PDC.")
    parser.add_argument("--use_gpdc", action="store_true", help="Use gPDC normalization.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.input_npy is not None:
        bold = np.load(args.input_npy)
        fs_from_loader = None
        if bold.ndim != 3:
            raise ValueError("--input_npy must contain an array of shape (num_subjects, num_rois, num_time_points)")
    else:
        bold, fs_from_loader = load_bold_data(args.dataset)

    N, R, T = bold.shape

    fs = _resolve_fs(args.fs, args.tr, fs_from_loader, args.method)

    start_idx, end_idx = _clamp_range(args.start, args.end, N)

    out_dir = _ensure_output_dir(Path(args.output_dir), args.dataset, args.method)

    print(f"Total subjects: {N}; processing indices [{start_idx}..{end_idx}] inclusive.")
    print(f"Dataset: {args.dataset}; Method: {args.method}; Output dir: {out_dir}")

    num_done = 0
    num_skipped = 0
    for s in range(start_idx, end_idx + 1):
        X = bold[s]  # (R, T)

        if args.method != "gc_fromSL":
            out_path = out_dir / f"{args.dataset}_{args.method}_subj_{s:06d}.npy"
            if out_path.exists() and not args.overwrite:
                num_skipped += 1
                continue

            fc = _compute_single_subject_fc(args.method, X, fs, args, dataset=args.dataset)

            _save_fc_matrix(out_dir, args.dataset, args.method, s, fc, overwrite=args.overwrite)
            num_done += 1

            print(f"Saved: {out_path}  (done={num_done}, skipped={num_skipped})")
        else:
            out_path_1 = out_dir / f"{args.dataset}_{args.method + '_F'}_subj_{s:06d}.npy"
            out_path_2 = out_dir / f"{args.dataset}_{args.method + '_P'}_subj_{s:06d}.npy"
            if out_path_1.exists() and out_path_2.exists() and not args.overwrite:
                num_skipped += 1
                continue

            F_mat, P_mat = _compute_single_subject_fc(args.method, X, fs, args, dataset=args.dataset)
            _save_fc_matrix(out_dir, args.dataset, args.method + "_F", s, F_mat, overwrite=args.overwrite)
            _save_fc_matrix(out_dir, args.dataset, args.method + "_P", s, P_mat, overwrite=args.overwrite)
            num_done += 1

            print(f"Saved: {out_path_1}")
            print(f"Saved: {out_path_2}")
            print(f"Done={num_done}, skipped={num_skipped}")

    print(f"Completed. Newly processed: {num_done}, skipped existing: {num_skipped}.")


if __name__ == "__main__":
    main()


