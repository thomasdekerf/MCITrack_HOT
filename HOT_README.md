# Hyperspectral Dataset Evaluation

This repository includes a utility for evaluating a hyperspectral object tracking dataset. Each sequence
should be stored in its own folder with numerically ordered frame images and a `groundtruth_rect.txt`
file containing `x y w h` annotations per line.

```
HOT_ROOT/
  sequence_1/
    1.png
    2.png
    ...
    groundtruth_rect.txt
  sequence_2/
    ...
```

## Running evaluation
1. Set the path in `lib/test/evaluation/local.py`:
   ```python
   settings.hot_path = '/path/to/HOT_ROOT'
   ```
2. Run the tracker and evaluation:
   ```bash
   python tracking/evaluate_hot.py mcitrack mcitrack_b224
   ```

Outputs for each sequence are saved under `tracking_results/mcitrack/hot/` and include:

- `sequence.mp4` – video with predicted (red) and ground-truth (green) boxes.
- `sequence.csv` – predicted bounding boxes.
- `metrics.json` – dataset precision@20 and success AUC values.

## Inference on unlabeled sequences

For running the tracker on a folder that only provides the first bounding box,
use `tracking/infer_folder.py`. The folder should contain the frame images and
an `init_rect.txt` file with the initial `x y w h` box:

```bash
python tracking/infer_folder.py mcitrack mcitrack_b224 --seq_dir /path/to/vis-worker \
    --output_csv vis-worker.csv
```

The script initializes the tracker from `init_rect.txt` and writes all predicted
boxes to the specified CSV file in the format `id,x,y,w,h`.
