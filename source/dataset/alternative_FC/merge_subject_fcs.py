import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np


SUBJ_REGEX = re.compile(r"_subj_(\d+)")
GC_FROM_SL_REGEX = re.compile(r"_gc_fromSL_(F|P)_subj_(\d+)", re.IGNORECASE)


def find_subject_files(input_dir: Path) -> List[Path]:
    files = sorted([p for p in input_dir.glob("*.npy") if p.is_file()])
    return files


def extract_subject_index(path: Path) -> int:
    m2 = GC_FROM_SL_REGEX.search(path.stem)
    if m2:
        try:
            return int(m2.group(2))
        except Exception:
            return -1

    m = SUBJ_REGEX.search(path.stem)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def load_fc(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"File {path} does not contain a square 2D FC matrix.")
    return arr


def stack_fc_matrices(files: List[Path]) -> Tuple[np.ndarray, List[int], List[Path], List[Path]]:
    indexed: List[Tuple[int, Path]] = []
    for f in files:
        idx = extract_subject_index(f)
        if idx >= 0:
            indexed.append((idx, f))
    if not indexed:
        raise RuntimeError("No subject files with pattern *_subj_XXXXXX.npy found.")
    indexed.sort(key=lambda x: x[0])

    # Load first to determine shape
    first = load_fc(indexed[0][1])
    r = first.shape[0]
    stack = np.empty((len(indexed), r, r), dtype=first.dtype)
    stack[0] = first

    bad_files: List[Path] = []
    subject_indices: List[int] = [indexed[0][0]]
    used_files: List[Path] = [indexed[0][1]]

    for k, (subj_idx, path) in enumerate(indexed[1:], start=1):
        try:
            arr = load_fc(path)
            if arr.shape != (r, r):
                raise ValueError(f"Inconsistent shape {arr.shape} vs {(r, r)} in {path}")
            stack[k] = arr
            subject_indices.append(subj_idx)
            used_files.append(path)
        except Exception:
            bad_files.append(path)
            # Fallback: identity to preserve indexing alignment
            stack[k] = np.eye(r, dtype=stack.dtype)
            subject_indices.append(subj_idx)
            used_files.append(path)

    return stack, subject_indices, used_files, bad_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-subject FC .npy files into a stacked (N,R,R) array.")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing per-subject FC .npy files.")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name, used for output path and filename.")
    parser.add_argument("--fc_type", required=True, type=str, help="FC type (e.g., frequency, phase, gc, pdc, gc_fromSL).")
    parser.add_argument("--output_root", default="/local/scratch3/khan58/BrainNetworkTransformer/alternate_fc/new_fcs", type=str,
                        help="Root directory to place merged file under {dataset}/.")
    return parser.parse_args()


def split_gc_fromSL_groups(files: List[Path]) -> Dict[str, List[Path]]:
    groups = {"F": [], "P": [], "OTHER": []}
    for p in files:
        m = GC_FROM_SL_REGEX.search(p.stem)
        if m:
            tag = m.group(1).upper()
            if tag == "F":
                groups["F"].append(p)
            elif tag == "P":
                groups["P"].append(p)
        else:
            groups["OTHER"].append(p)
    return groups


def save_stack(out_dir: Path, dataset: str, fc_type: str, suffix: str, stack: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if suffix:
        out_path = out_dir / f"{dataset}_{fc_type}_{suffix}.npy"
    else:
        out_path = out_dir / f"{dataset}_{fc_type}.npy"
    np.save(out_path, stack)
    return out_path


def merge_and_report(files: List[Path], label: str, out_dir: Path, dataset: str, fc_type: str) -> None:
    if not files:
        print(f"[{label}] No matching files found for gc_fromSL_{label}. Skipped.")
        return

    stacked, subject_indices, used_files, bad_files = stack_fc_matrices(files)
    out_path = save_stack(out_dir, dataset, fc_type, label, stacked)

    print(f"[{label}] Saved merged FC: {out_path}")
    print(f"[{label}] Shape: {stacked.shape}")
    print(f"[{label}] Subjects merged: {len(subject_indices)} (min_idx={min(subject_indices)}, max_idx={max(subject_indices)})")
    if bad_files:
        print(f"[{label}] Warning: {len(bad_files)} files failed to load; filled with identity. Example: {bad_files[0]}")


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    files = find_subject_files(in_dir)
    if not files:
        raise RuntimeError(f"No .npy files found under {in_dir}")

    out_dir = Path(args.output_root) / args.dataset

    fc_type_lower = args.fc_type.lower()
    is_gc_fromSL = ("gc_fromSL" in fc_type_lower) or any("gc_fromSL" in part for part in in_dir.parts)

    if is_gc_fromSL:
        groups = split_gc_fromSL_groups(files)
        merge_and_report(groups.get("F", []), "F", out_dir, args.dataset, args.fc_type)
        merge_and_report(groups.get("P", []), "P", out_dir, args.dataset, args.fc_type)

        if groups.get("OTHER"):
            print(f"[INFO] {len(groups['OTHER'])} files did not match gc_fromSL F/P pattern and were ignored. Example: {groups['OTHER'][0]}")
    else:
        stacked, subject_indices, used_files, bad_files = stack_fc_matrices(files)
        out_path = save_stack(out_dir, args.dataset, args.fc_type, suffix="", stack=stacked)
        print(f"Saved merged FC: {out_path}")
        print(f"Shape: {stacked.shape}")
        print(f"Subjects merged: {len(subject_indices)} (min_idx={min(subject_indices)}, max_idx={max(subject_indices)})")
        if bad_files:
            print(f"Warning: {len(bad_files)} files failed to load; filled with identity. Example: {bad_files[0]}")


if __name__ == "__main__":
    main()


