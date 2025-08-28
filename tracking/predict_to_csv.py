#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
from tqdm import tqdm

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import run_dataset


def read_init_rect(path):
    """Read first bbox from init_rect.txt (x y w h), supports tab/space/comma separators."""
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        line = f.readline().strip()
    for sep in ['\t', ',', ' ']:
        if sep in line:
            parts = [p for p in line.split(sep) if p != '']
            break
    else:
        parts = [line]
    vals = list(map(float, parts[:4]))
    if len(vals) != 4:
        raise ValueError(f"Invalid init_rect in {path}: {line}")
    return np.array(vals, dtype=float)


def ensure_seq_init_from_file(seq, init_rect_filename):
    """
    If a sequence folder contains an init_rect file, replace the first GT box with it,
    so the tracker initializes from the provided starting point.
    """
    # Try both direct seq path and parent guesses
    candidates = [
        os.path.join(seq.path, init_rect_filename) if hasattr(seq, 'path') else None,
        os.path.join(os.path.dirname(seq.frames[0]), init_rect_filename) if seq.frames else None
    ]
    candidates = [p for p in candidates if p]

    for p in candidates:
        init_rect = read_init_rect(p)
        if init_rect is not None:
            # Force the dataset object to use this init
            if hasattr(seq, 'ground_truth_rect') and len(seq.ground_truth_rect) > 0:
                seq.ground_truth_rect[0] = init_rect
            return True
    return False


def find_pred_base(results_root, seq_name):
    """Some trackers write results as <root>/<seq>.* or <root>/hot/<seq>.* — support both."""
    base_direct = os.path.join(results_root, seq_name)
    base_with_ds = os.path.join(results_root, 'hot', seq_name)
    if os.path.isfile(base_direct + '.txt'):
        return base_direct
    return base_with_ds


def load_tracker_preds_as_xywh(txt_path):
    """
    Load tracker predictions in x,y,w,h per line.
    HOT trackers typically use tab-separated; we tolerate tab/space/comma.
    """
    # Try common delimiters
    for delim in ['\t', ',', ' ']:
        try:
            arr = np.loadtxt(txt_path, delimiter=delim)
            if arr.ndim == 1 and arr.size == 4:
                arr = arr.reshape(1, 4)
            if arr.shape[1] == 4:
                return arr.astype(float)
        except Exception:
            continue
    # Fallback: raw parsing
    rows = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = [p for p in line.strip().replace(',', ' ').replace('\t', ' ').split(' ') if p]
            if len(parts) < 4:
                continue
            rows.append(list(map(float, parts[:4])))
    if not rows:
        raise RuntimeError(f"Could not parse predictions: {txt_path}")
    return np.array(rows, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Run tracker on HOT and produce a single competition-style CSV.")
    parser.add_argument('tracker_name', type=str, help='Tracking method name')
    parser.add_argument('tracker_param', type=str, help='Tracker parameter name')
    parser.add_argument('--output_csv', type=str, default='submission.csv',
                        help='Path to the single output CSV (default: submission.csv)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Optional single sequence name to run/aggregate')
    parser.add_argument('--prefix', type=str, default='vis',
                        help='ID prefix (e.g., vis -> vis-<sequence>_<idx>)')
    parser.add_argument('--start_index', type=int, default=1,
                        help='Frame index base for IDs (1 for _1, 0 for _0, ...)')
    parser.add_argument('--init_rect_filename', type=str, default='init_rect.txt',
                        help='Filename of the starting bbox in each sequence directory')
    parser.add_argument('--skip_run', action='store_true',
                        help='Reuse existing predictions; do not re-run the tracker')
    parser.add_argument('--threads', type=int, default=0, help='run_dataset threads')
    parser.add_argument('--num_gpus', type=int, default=1, help='GPUs for run_dataset')
    parser.add_argument('--debug', type=int, default=0, help='Debug level for run_dataset')
    args = parser.parse_args()

    # Load dataset
    dataset = get_dataset('hot')
    if args.sequence is not None:
        dataset = [seq for seq in dataset if seq.name == args.sequence]
        if not dataset:
            raise ValueError(f"Sequence '{args.sequence}' not found.")

    # Prepare tracker
    tracker = Tracker(args.tracker_name, args.tracker_param, 'hot')

    # Ensure init from init_rect.txt when available
    for seq in dataset:
        ensure_seq_init_from_file(seq, args.init_rect_filename)

    # Run tracker unless we reuse existing outputs
    if not args.skip_run:
        run_dataset(dataset, [tracker], args.debug, threads=args.threads, num_gpus=args.num_gpus)

    results_root = tracker.results_dir

    # Aggregate all predictions into a single CSV
    # Format: id,x,y,w,h  with id like "<prefix>-<sequence>_<frame_idx>"
    rows = []
    for seq in tqdm(dataset, desc="Collecting predictions"):
        base_path = find_pred_base(results_root, seq.name)
        pred_txt = base_path + '.txt'
        if not os.path.isfile(pred_txt):
            raise FileNotFoundError(f"Missing predictions: {pred_txt}")

        preds = load_tracker_preds_as_xywh(pred_txt)
        n_frames = len(preds)

        # build IDs
        for i in range(n_frames):
            idx = i + args.start_index
            id_str = f"{args.prefix}-{seq.name}_{idx}"
            x, y, w, h = preds[i].tolist()
            rows.append((id_str, x, y, w, h))

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'w', 'h'])
        for r in rows:
            writer.writerow(r)

    print(f"Submission written to: {os.path.abspath(args.output_csv)}")
    print(f"Total rows: {len(rows)}")


if __name__ == '__main__':
    main()
