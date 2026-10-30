import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data

import h5py


class GeoSegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path,
        split,
        point_limit=None,
    ):
        super(GeoSegDataset, self).__init__()

        self.h5_path = h5_path
        self.split = split
        self.point_limit = point_limit

        with h5py.File(h5_path) as f:
            self.keys = list(f[split].keys())

    def __len__(self):
        return len(self.keys)

    def _load_data(self, key):
        with h5py.File(self.h5_path) as f:
            points_object = np.asarray(f[self.split][key]["object_points"], dtype=np.float32)
            points_scene = np.asarray(f[self.split][key]["scene_points"], dtype=np.float32)
            labels = np.asarray(f[self.split][key]["labels"], dtype=np.int16)

        return points_object, points_scene, labels

    def __getitem__(self, index):
        data_dict = {}

        key = self.keys[index]

        data_dict['scene_name'] = f"{self.split}_{key}]"

        points_object, points_scene, labels = self._load_data(key)

        data_dict['points_object'] = points_object
        data_dict['points_scene'] = points_scene
        data_dict['labels'] = labels

        return data_dict
