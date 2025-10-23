import argparse
import os
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge subject-level node feature matrices into a single array (N, R, D) and save as {dataset}_{feature}.npy\n"
            "Input directory should contain per-subject .npy files with shape (R, D)."
        )
    )
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing subject-level node feature .npy files.")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name, e.g., ABIDE, ABCD, HCP, PNC.")
    parser.add_argument("--feature", required=True, type=str, help="Node feature type, e.g., stat or psd.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="/local/scratch3/khan58/BrainNetworkTransformer/alternate_node_feature/new_node_features",
        help="Root output directory; merged file will be saved under {output_root}/{dataset}/",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional filename pattern to filter files (substring match). Default: include all .npy files.",
    )
    return parser.parse_args()


def _extract_subject_index(filename: str) -> int:
    """Try to extract subject index like *_subj_000123.npy -> 123 for numeric sorting; fallback to 0."""
    m = re.search(r"_subj_(\d+)\.npy$", filename)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _list_feature_files(input_dir: Path, pattern: str | None) -> List[Path]:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".npy"]
    if pattern:
        files = [p for p in files if pattern in p.name]
    files.sort(key=lambda p: (_extract_subject_index(p.name), p.name))
    return files


def _load_and_stack(files: List[Path]) -> np.ndarray:
    if len(files) == 0:
        raise ValueError("No .npy files found to merge.")

    # Probe first file for shape
    first = np.load(files[0])
    if first.ndim != 2:
        raise ValueError(f"Expected each file to be 2D (R, D); got shape {first.shape} in {files[0]}")
    R, D = first.shape

    stack = np.zeros((len(files), R, D), dtype=first.dtype)
    stack[0] = first

    for i, path in enumerate(files[1:], start=1):
        arr = np.load(path)
        if arr.shape != (R, D):
            raise ValueError(f"Shape mismatch: {path} has shape {arr.shape}, expected {(R, D)}")
        stack[i] = arr

    return stack


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = _list_feature_files(input_dir, args.pattern)
    print(f"Found {len(files)} files in {input_dir}")

    merged = _load_and_stack(files)
    print(f"Merged shape: {merged.shape}  (num_subjects, num_rois, feature_dim)")

    out_dir = Path(args.output_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.feature}.npy"

    np.save(out_path, merged)
    print(f"Saved merged file: {out_path}")


if __name__ == "__main__":
    main()
