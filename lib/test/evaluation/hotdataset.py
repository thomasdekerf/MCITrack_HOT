import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import re
class HOTDataset(BaseDataset):
    """Dataset for generic hyperspectral object tracking sequences.

    Each sequence folder should contain the frame images and a
    ``groundtruth_rect.txt`` file with one ``x y w h`` annotation per line.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.hot_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _get_sequence_list(self):
        if not os.path.isdir(self.base_path):
            return []
        seqs = [d for d in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, d))]
        seqs.sort()
        return seqs

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name)
        frame_files = [f for f in os.listdir(seq_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        # --- robust sort: use trailing number like ...0001, ...0002, etc.
        def frame_key(fname: str):
            base = os.path.splitext(fname)[0]
            m = re.search(r'(\d+)$', base)  # last number at end of name
            # sort primarily by the numeric suffix (if any), then by full basename to stabilize order
            return (int(m.group(1)) if m else float('inf'), base)

        frame_files = sorted(frame_files, key=frame_key)
        frames = [os.path.join(seq_path, f) for f in frame_files]

        # anno_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        anno_path = os.path.join(seq_path, 'init_rect.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        ground_truth_rect = ground_truth_rect.reshape(-1, 4)

        # --- optional safety: match #frames with #annotations
        if len(frames) == 0 or len(ground_truth_rect) == 0:
            raise RuntimeError(f"No frames/annotations found in {seq_path}")

        if len(frames) != len(ground_truth_rect):
            if len(ground_truth_rect) == 1 and len(frames) > 1:
                print(
                    f"[HOTDataset] Warning: {sequence_name} has {len(frames)} frames but 1 anno. "
                    "Using the first annotation for initialization and keeping all frames.")
                # Repeat the first annotation so that Sequence has per-frame boxes
                first = ground_truth_rect[0]
                ground_truth_rect = np.vstack([first for _ in range(len(frames))])
            else:
                n = min(len(frames), len(ground_truth_rect))
                print(
                    f"[HOTDataset] Warning: {sequence_name} has {len(frames)} frames but {len(ground_truth_rect)} annos. Truncating to {n}.")
                frames = frames[:n]
                ground_truth_rect = ground_truth_rect[:n]

        return Sequence(sequence_name, frames, 'hot', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)
