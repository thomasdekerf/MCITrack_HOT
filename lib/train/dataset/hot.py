import os
import re
import torch
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class HOT(BaseVideoDataset):
    """Minimal dataset wrapper for hyperspectral object tracking (HOT).

    Assumes each sequence directory contains the frame images and either
    ``groundtruth_rect.txt`` or ``init_rect.txt`` with one ``x y w h`` box per line.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader):
        root = env_settings().hot_dir if root is None else root
        super().__init__('HOT', root, image_loader)
        self.sequence_list = [d for d in sorted(os.listdir(self.root))
                              if os.path.isdir(os.path.join(self.root, d))]

    def get_name(self):
        return 'hot'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_frame_files(self, seq_path):
        frame_files = [f for f in os.listdir(seq_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        def frame_key(fname: str):
            base = os.path.splitext(fname)[0]
            m = re.search(r'(\d+)$', base)
            return (int(m.group(1)) if m else float('inf'), base)

        return sorted(frame_files, key=frame_key)

    def _read_bb_anno(self, seq_path):
        gt_path = None
        for name in ['groundtruth_rect.txt', 'init_rect.txt']:
            p = os.path.join(seq_path, name)
            if os.path.isfile(p):
                gt_path = p
                break
        if gt_path is None:
            raise FileNotFoundError(f'No annotation file found in {seq_path}')
        gt = pandas.read_csv(gt_path, sep='\t|,| ', header=None, engine='python', dtype=float).values
        gt = gt.reshape(-1, 4)
        return torch.tensor(gt, dtype=torch.float32)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.ones(len(bbox), dtype=torch.bool)
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_files = self._get_frame_files(seq_path)
        frame_list = [self.image_loader(os.path.join(seq_path, frame_files[f_id])) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {k: [v[f_id].clone() for f_id in frame_ids] for k, v in anno.items()}
        object_meta = OrderedDict({'object_class_name': None})
        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        return {k: [v[f_id].clone() for f_id in frame_ids] for k, v in anno.items()}
