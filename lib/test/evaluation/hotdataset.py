import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

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
        frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        frames = [os.path.join(seq_path, f) for f in frame_files]

        anno_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        ground_truth_rect = ground_truth_rect.reshape(-1, 4)

        return Sequence(sequence_name, frames, 'hot', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)
