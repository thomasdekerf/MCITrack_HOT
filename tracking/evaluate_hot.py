import os
import cv2
import json
import argparse
import numpy as np
import torch
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation.environment import env_settings


def main():
    parser = argparse.ArgumentParser(description='Evaluate tracker on HOT dataset')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file')
    parser.add_argument('--sequence', type=str, default=None, help='Optional sequence name')
    parser.add_argument('--debug', type=int, default=0, help='Debug level')
    args = parser.parse_args()

    dataset = get_dataset('hot')
    if args.sequence is not None:
        dataset = [seq for seq in dataset if seq.name == args.sequence]

    tracker = Tracker(args.tracker_name, args.tracker_param, 'hot')
    run_dataset(dataset, [tracker], args.debug, threads=0, num_gpus=1)

    settings = env_settings()
    results_root = tracker.results_dir

    for seq in dataset:
        base_path = os.path.join(results_root, 'hot', seq.name)
        pred_path = base_path + '.txt'
        preds = np.loadtxt(pred_path, delimiter='\t')
        csv_path = base_path + '.csv'
        np.savetxt(csv_path, preds, delimiter=',', fmt='%.2f')

        video_path = base_path + '.mp4'
        first_frame = cv2.imread(seq.frames[0])
        h, w = first_frame.shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        gt = np.array(seq.ground_truth_rect, dtype=int)
        for i, frame_path in enumerate(seq.frames):
            frame = cv2.imread(frame_path)
            bb_pred = preds[i].astype(int)
            bb_gt = gt[i]
            cv2.rectangle(frame, (bb_pred[0], bb_pred[1]),
                          (bb_pred[0] + bb_pred[2], bb_pred[1] + bb_pred[3]), (0, 0, 255), 2)
            cv2.rectangle(frame, (bb_gt[0], bb_gt[1]),
                          (bb_gt[0] + bb_gt[2], bb_gt[1] + bb_gt[3]), (0, 255, 0), 2)
            writer.write(frame)
        writer.release()

    eval_data = extract_results([tracker], dataset, 'hot_eval')
    precision_curve = torch.tensor(eval_data['ave_success_rate_plot_center']).mean(0)
    thresholds = torch.tensor(eval_data['threshold_set_center'])
    dp20 = precision_curve[thresholds == 20].item()
    success_curve = torch.tensor(eval_data['ave_success_rate_plot_overlap']).mean(0)
    overlaps = torch.tensor(eval_data['threshold_set_overlap'])
    auc = torch.trapz(success_curve, overlaps).item()

    metrics = {
        'precision@20': dp20,
        'AUC': auc
    }
    metrics_path = os.path.join(results_root, 'hot', 'metrics.json')
    with open(metrics_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(metrics)


if __name__ == '__main__':
    main()
