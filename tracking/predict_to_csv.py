#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import cv2
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
    candidates = [
        os.path.join(seq.path, init_rect_filename) if hasattr(seq, 'path') else None,
        os.path.join(os.path.dirname(seq.frames[0]), init_rect_filename) if getattr(seq, 'frames', None) else None
    ]
    candidates = [p for p in candidates if p]

    for p in candidates:
        init_rect = read_init_rect(p)
        if init_rect is not None:
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


def parse_submission_csv(csv_path):
    """
    Read a single competition-style CSV: columns id,x,y,w,h with id like <seq>_<idx>.
    Returns: dict mapping seq -> dict(frame_idx -> (x,y,w,h))
    """
    mapping = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_str = row['id']
            # split from the right to avoid underscores in seq names breaking
            if '_' not in id_str:
                continue
            seq, idx_str = id_str.rsplit('_', 1)
            try:
                idx = int(idx_str)
            except ValueError:
                # tolerate non-int (e.g., 0001); try to parse as int anyway
                idx = int(float(idx_str))
            x = float(row['x']); y = float(row['y']); w = float(row['w']); h = float(row['h'])
            mapping.setdefault(seq, {})[idx] = (x, y, w, h)
    return mapping


def draw_bbox(frame_bgr, bbox, color=(0, 255, 0), thickness=2, label=None):
    x, y, w, h = bbox
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(frame_bgr, p1, p2, color, thickness)
    if label is not None:
        cv2.putText(frame_bgr, str(label), (p1[0], max(0, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def write_sequence_video(seq, preds_xywh, out_path, fps=24, start_index=1):
    """
    seq: dataset sequence object with seq.frames (list of image paths)
    preds_xywh: np.ndarray shape [N,4] aligned to frames order
    """
    if len(seq.frames) == 0:
        raise ValueError(f"Sequence {seq.name} has no frames.")
    # Read first frame to set video size
    first = cv2.imread(seq.frames[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first frame for {seq.name}: {seq.frames[0]}")
    H, W = first.shape[:2]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    n = min(len(seq.frames), len(preds_xywh))
    for i in range(n):
        frame = cv2.imread(seq.frames[i], cv2.IMREAD_COLOR)
        if frame is None:
            # Write black frame if missing
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        bbox = preds_xywh[i]
        draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2, label=f"{seq.name}_{i + start_index}")
        writer.write(frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="Run tracker on HOT, optionally aggregate CSV, and make per-sequence videos with bboxes.")
    parser.add_argument('tracker_name', type=str, help='Tracking method name')
    parser.add_argument('tracker_param', type=str, help='Tracker parameter name')

    # IO controls
    parser.add_argument('--output_csv', type=str, default='submission.csv',
                        help='Path to the output CSV when aggregating from tracker results')
    parser.add_argument('--from_csv', type=str, default=None,
                        help='Use an existing submission CSV (skip aggregation from tracker .txt)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Optional single sequence name to run/aggregate/video')

    # Running & aggregation controls
    parser.add_argument('--start_index', type=int, default=1,
                        help='Frame index base for IDs (1 for _1, 0 for _0, ...)')
    parser.add_argument('--init_rect_filename', type=str, default='init_rect.txt',
                        help='Filename of the starting bbox in each sequence directory')
    parser.add_argument('--skip_run', action='store_true',
                        help='Reuse existing predictions; do not re-run the tracker')
    parser.add_argument('--threads', type=int, default=0, help='run_dataset threads')
    parser.add_argument('--num_gpus', type=int, default=1, help='GPUs for run_dataset')
    parser.add_argument('--debug', type=int, default=0, help='Debug level for run_dataset')

    # Video options
    parser.add_argument('--make_videos', action='store_true', help='Create a clip per sequence with bboxes')
    parser.add_argument('--video_dir', type=str, default='videos', help='Directory to write sequence MP4s')
    parser.add_argument('--fps', type=int, default=24, help='Video framerate')

    args = parser.parse_args()

    # Load dataset
    dataset = get_dataset('hot')
    if args.sequence is not None:
        dataset = [seq for seq in dataset if seq.name == args.sequence]
        if not dataset:
            raise ValueError(f"Sequence '{args.sequence}' not found.")

    # Using an existing CSV: skip any tracker work, just load predictions & optionally make videos.
    if args.from_csv is not None:
        pred_map = parse_submission_csv(args.from_csv)

        # Optionally filter sequences not in CSV
        dataset = [seq for seq in dataset if seq.name in pred_map]

        # Make videos or just validate
        if args.make_videos:
            os.makedirs(args.video_dir, exist_ok=True)
            for seq in tqdm(dataset, desc="Making videos from CSV"):
                # Build per-frame predictions aligned with seq.frames
                n = len(seq.frames)
                preds_seq = []
                for i in range(n):
                    idx = i + args.start_index
                    if seq.name not in pred_map or idx not in pred_map[seq.name]:
                        raise KeyError(f"Missing prediction for {seq.name}_{idx} in CSV.")
                    preds_seq.append(pred_map[seq.name][idx])
                preds_seq = np.asarray(preds_seq, dtype=float)

                out_path = os.path.join(args.video_dir, f"{seq.name}.mp4")
                write_sequence_video(seq, preds_seq, out_path, fps=args.fps, start_index=args.start_index)
            print(f"Videos written to: {os.path.abspath(args.video_dir)}")
        else:
            print("Loaded predictions from CSV; no videos requested. Nothing else to do.")
        return

    # Otherwise: prepare tracker route (optional run + aggregation from tracker .txt)
    tracker = Tracker(args.tracker_name, args.tracker_param, 'hot')

    # Ensure init from init_rect.txt when available
    for seq in dataset:
        ensure_seq_init_from_file(seq, args.init_rect_filename)

    # Run tracker unless we reuse existing outputs
    if not args.skip_run:
        run_dataset(dataset, [tracker], args.debug, threads=args.threads, num_gpus=args.num_gpus)

    results_root = tracker.results_dir

    # Aggregate predictions into a single CSV
    rows = []
    seq_to_preds = {}  # keep for video making
    for seq in tqdm(dataset, desc="Collecting predictions"):
        base_path = find_pred_base(results_root, seq.name)
        pred_txt = base_path + '.txt'
        if not os.path.isfile(pred_txt):
            raise FileNotFoundError(f"Missing predictions: {pred_txt}")

        preds = load_tracker_preds_as_xywh(pred_txt)
        seq_to_preds[seq.name] = preds  # cache for videos

        # CSV rows
        for i in range(len(preds)):
            idx = i + args.start_index
            id_str = f"{seq.name}_{idx}"
            x, y, w, h = preds[i].tolist()
            rows.append((id_str, x, y, w, h))

    # Write CSV
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'x', 'y', 'w', 'h'])
            writer.writerows(rows)
        print(f"Submission written to: {os.path.abspath(args.output_csv)}")
        print(f"Total rows: {len(rows)}")

    # Make videos if requested
    if args.make_videos:
        os.makedirs(args.video_dir, exist_ok=True)
        for seq in tqdm(dataset, desc="Making videos from tracker outputs"):
            preds_seq = seq_to_preds[seq.name]
            out_path = os.path.join(args.video_dir, f"{seq.name}.mp4")
            write_sequence_video(seq, preds_seq, out_path, fps=args.fps, start_index=args.start_index)
        print(f"Videos written to: {os.path.abspath(args.video_dir)}")


if __name__ == '__main__':
    main()
