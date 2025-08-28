import os
import cv2
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation.environment import env_settings


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_preds_txt(txt_path):
    # tracker outputs are typically tab-separated x,y,w,h per line
    return np.loadtxt(txt_path, delimiter='\t')


def save_csv_from_preds(preds, csv_path):
    # Save as comma-separated floats with 2 decimals
    np.savetxt(csv_path, preds, delimiter=',', fmt='%.2f')


def draw_bboxes(frame, bb_pred, bb_gt, pred_color=(0, 0, 255), gt_color=(0, 255, 0), thickness=2):
    # bb_* are [x, y, w, h]
    x, y, w, h = map(int, bb_pred)
    cv2.rectangle(frame, (x, y), (x + w, y + h), pred_color, thickness)
    xg, yg, wg, hg = map(int, bb_gt)
    cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), gt_color, thickness)
    return frame


def make_video_from_csv(csv_path, frames, gts, out_path, fps=30):
    preds = np.loadtxt(csv_path, delimiter=',')
    n = min(len(preds), len(frames), len(gts))
    if n == 0:
        raise RuntimeError(f"No frames to render for video: {out_path}")

    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for i in tqdm(range(n), desc=f"Video {os.path.basename(out_path)}", leave=False):
        frame = cv2.imread(frames[i])
        frame = draw_bboxes(frame, preds[i], gts[i])
        writer.write(frame)

    writer.release()


def robust_curve_mean(curve_tensor):
    """
    Accepts tensors with shapes like [N_seq, 51], [1, 51], [51], or extra leading dims.
    Returns a 1D vector over thresholds.
    """
    t = torch.as_tensor(curve_tensor)
    if t.dim() == 0:
        return t.reshape(1)
    if t.dim() == 1:
        return t.reshape(-1)
    # average across all leading dims except the last (threshold dimension)
    lead_dims = tuple(range(t.dim() - 1))
    return t.mean(dim=lead_dims).reshape(-1)


def robust_per_seq(curve_tensor):
    """
    Ensures a per-sequence matrix [N_seq, T] where T is thresholds length.
    If input is [T], we interpret it as 1 sequence.
    """
    t = torch.as_tensor(curve_tensor)
    if t.dim() == 1:
        return t.unsqueeze(0)  # [1, T]
    # collapse any extra leading dims into N_seq, keep last as T
    T = t.size(-1)
    N = int(t.numel() // T)
    return t.reshape(N, T)


def compute_metrics_per_sequence(eval_data):
    # Extract curves and thresholds
    pc = torch.as_tensor(eval_data['ave_success_rate_plot_center'])   # precision over center-thresholds
    sc = torch.as_tensor(eval_data['ave_success_rate_plot_overlap'])  # success over IoU thresholds
    thr_center = torch.as_tensor(eval_data['threshold_set_center']).reshape(-1)
    thr_overlap = torch.as_tensor(eval_data['threshold_set_overlap']).reshape(-1)

    # Standardize to [N_seq, T]
    pc_mat = robust_per_seq(pc)     # [N_seq, Tc]
    sc_mat = robust_per_seq(sc)     # [N_seq, To]

    # dp20 per sequence
    idx20 = int((thr_center - 20.0).abs().argmin().item())
    dp20_seq = pc_mat[:, idx20]     # [N_seq]

    # AUC per sequence (trapz over IoU thresholds)
    auc_seq = []
    for i in range(sc_mat.size(0)):
        auc_val = torch.trapz(sc_mat[i].reshape(-1), thr_overlap).item()
        auc_seq.append(auc_val)
    auc_seq = torch.tensor(auc_seq)

    return dp20_seq, auc_seq, thr_center, thr_overlap


def plot_bars(names, values, ylabel, out_path, title=None, rotation=60):
    plt.figure(figsize=(max(8, 0.25 * len(names)), 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(names)), names, rotation=rotation, ha='right')
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate tracker on HOT dataset')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file')
    parser.add_argument('--sequence', type=str, default=None, help='Optional single sequence to process')
    parser.add_argument('--debug', type=int, default=0, help='Debug level')
    parser.add_argument('--skip_run', action='store_true',
                        help='Skip running the tracker; reuse existing predictions')
    parser.add_argument('--skip_vid', action='store_true', help = 'skips video generation')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output videos')
    parser.add_argument('--threads', type=int, default=0, help='run_dataset threads (0 = main thread)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for run_dataset')
    args = parser.parse_args()

    # Dataset selection
    dataset = get_dataset('hot')
    if args.sequence is not None:
        dataset = [seq for seq in dataset if seq.name == args.sequence]
        if not dataset:
            raise ValueError(f"Sequence '{args.sequence}' not found in HOT dataset.")

    tracker = Tracker(args.tracker_name, args.tracker_param, 'hot')

    # 1) Optionally run the tracker to produce .txt predictions
    if not args.skip_run:
        run_dataset(dataset, [tracker], args.debug, threads=args.threads, num_gpus=args.num_gpus)

    # 2) Post-processing: CSV + Video
    settings = env_settings()  # not strictly needed, but keeps compatibility
    results_root = tracker.results_dir  # e.g., .../results/<tracker_name>/<param_name>

    # Some trackers place files under results_root/<dataset_name>/<seq>, others directly /<seq>
    # We'll try direct path first; if missing, try including 'hot'.
    def seq_base_path(seq_name):
        base_direct = os.path.join(results_root, seq_name)
        base_with_ds = os.path.join(results_root, 'hot', seq_name)
        if os.path.isfile(base_direct + '.txt'):
            return base_direct
        return base_with_ds

    for seq in tqdm(dataset, desc="Post-processing sequences"):
        base_path = seq_base_path(seq.name)
        txt_path = base_path + '.txt'
        csv_path = base_path + '.csv'
        mp4_path = base_path + '.mp4'

        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Missing prediction file: {txt_path}")

        preds = load_preds_txt(txt_path)
        save_csv_from_preds(preds, csv_path)

        # Build video from CSV predictions
        gt = np.array(seq.ground_truth_rect, dtype=int)
        if not args.skip_vid:
            make_video_from_csv(csv_path, seq.frames, gt, mp4_path, fps=args.fps)

    # 3) Metrics (per-sequence + overall)
    eval_data = extract_results([tracker], dataset, 'hot_eval')

    dp20_seq, auc_seq, thr_center, thr_overlap = compute_metrics_per_sequence(eval_data)
    seq_names = [seq.name for seq in dataset]

    dp20_avg = float(dp20_seq.mean().item()) if dp20_seq.numel() > 0 else 0.0
    auc_avg = float(auc_seq.mean().item()) if auc_seq.numel() > 0 else 0.0

    # 4) Plots
    plots_dir = ensure_dir(os.path.join(results_root, 'plots'))
    plot_bars(seq_names, [float(v) for v in dp20_seq.tolist()],
              ylabel='Precision @ 20px', out_path=os.path.join(plots_dir, 'dp20_by_sequence.png'),
              title='dp20 by sequence')
    plot_bars(seq_names, [float(v) for v in auc_seq.tolist()],
              ylabel='AUC (success over IoU thresholds)', out_path=os.path.join(plots_dir, 'auc_by_sequence.png'),
              title='AUC by sequence')

    # 5) JSON summary (overall + per-sequence)
    metrics = {
        "overall": {
            "num_sequences": len(seq_names),
            "precision@20_mean": dp20_avg,
            "AUC_mean": auc_avg
        },
        "per_sequence": [
            {"sequence": n, "precision@20": float(dp20_seq[i].item()), "AUC": float(auc_seq[i].item())}
            for i, n in enumerate(seq_names)
        ]
    }

    metrics_path = os.path.join(results_root, 'hot_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone.\n"
          f"- Plots: {plots_dir}\n"
          f"- Metrics JSON: {metrics_path}\n"
          f"- CSV/MP4 live next to each sequence under: {results_root}\n")


if __name__ == '__main__':
    main()
