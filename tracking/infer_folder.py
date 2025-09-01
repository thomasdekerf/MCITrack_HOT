#!/usr/bin/env python3
"""Run tracker inference on a directory of frames and export predictions to CSV.

This script is intended for sequences that have no per-frame annotations. The
directory must contain the frame images and an ``init_rect.txt`` file with the
starting bounding box in ``x y w h`` format. The tracker is initialized with
this box and run over all frames. The resulting predictions are written to a
CSV file of the form ``id,x,y,w,h`` where ``id`` combines the sequence name and
frame index (e.g. ``vis-worker_1``).
"""

import argparse
import csv
import os
import re
from typing import List

import numpy as np

from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.data import Sequence


def _sorted_frames(seq_dir: str) -> List[str]:
    """Return a numerically sorted list of frame file paths."""
    frame_files = [f for f in os.listdir(seq_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    def frame_key(fname: str):
        base = os.path.splitext(fname)[0]
        m = re.search(r'(\d+)$', base)
        return (int(m.group(1)) if m else float('inf'), base)

    frame_files.sort(key=frame_key)
    return [os.path.join(seq_dir, f) for f in frame_files]


def _read_init_rect(seq_dir: str, filename: str) -> np.ndarray:
    """Read ``x y w h`` from ``init_rect.txt`` inside ``seq_dir``."""
    path = os.path.join(seq_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing init rect file: {path}")
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


def main():
    parser = argparse.ArgumentParser(description="Run tracker on a folder of frames and export predictions to CSV.")
    parser.add_argument('tracker_name', help='Name of tracking method')
    parser.add_argument('tracker_param', help='Tracker parameter name')
    parser.add_argument('--seq_dir', required=True, help='Path to sequence directory with frames and init_rect.txt')
    parser.add_argument('--output_csv', default='predictions.csv', help='Output CSV path')
    parser.add_argument('--id_prefix', default=None,
                        help='Optional prefix for the id column (default: sequence folder name)')
    parser.add_argument('--init_rect_filename', default='init_rect.txt', help='Init bbox filename')
    parser.add_argument('--start_index', type=int, default=1, help='Frame index start for ids')
    parser.add_argument('--debug', type=int, default=0, help='Debug level for tracker')
    args = parser.parse_args()

    seq_dir = os.path.abspath(args.seq_dir)
    frames = _sorted_frames(seq_dir)
    if not frames:
        raise RuntimeError(f"No frame images found in {seq_dir}")

    init_bbox = _read_init_rect(seq_dir, args.init_rect_filename)
    gt = np.zeros((len(frames), 4), dtype=float)
    gt[0] = init_bbox

    seq_name = args.id_prefix or os.path.basename(seq_dir.rstrip('/'))
    seq = Sequence(seq_name, frames, 'hot', gt)

    tracker = Tracker(args.tracker_name, args.tracker_param, 'hot')
    output = tracker.run_sequence(seq, debug=args.debug)
    preds = np.array(output['target_bbox'])

    rows = []
    for i, (x, y, w, h) in enumerate(preds, start=args.start_index):
        rows.append((f"{seq_name}_{i}", x, y, w, h))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'w', 'h'])
        writer.writerows(rows)

    print(f"Predictions saved to: {os.path.abspath(args.output_csv)}")
    print(f"Total frames: {len(rows)}")


if __name__ == '__main__':
    main()
