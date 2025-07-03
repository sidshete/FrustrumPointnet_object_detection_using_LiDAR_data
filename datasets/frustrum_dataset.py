import os
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.preprocess import preprocess_frame

class FrustumDataset(Dataset):
    def __init__(self, lidar_dir, label_dir, calib_dir, split='train', npoints=1024, type_filter='Car'):
        self.lidar_dir = lidar_dir
        self.label_dir = label_dir
        self.calib_dir = calib_dir
        self.npoints = npoints
        self.type_filter = type_filter
        self.sample_list = []

        self.load_split(split)

    def load_split(self, split):
        split_file = os.path.join(self.lidar_dir, f'{split}_split.txt')
        with open(split_file, 'r') as f:
            frame_ids = [line.strip() for line in f.readlines()]
        
        for frame_id in frame_ids:
            lidar_path = os.path.join(self.lidar_dir, frame_id + '.bin')
            label_path = os.path.join(self.label_dir, frame_id + '.txt')
            calib_path = os.path.join(self.calib_dir, frame_id + '.txt')

            frustum_data = preprocess_frame(lidar_path, label_path, calib_path)

            for entry in frustum_data:
                self.sample_list.append(entry)

    def random_rotation(self, pc):
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)
        rot_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        return pc @ rot_matrix.T

    def random_scaling(self, pc, scale_range=(0.95, 1.05)):
        scale = np.random.uniform(*scale_range)
        return pc * scale

    def random_translation(self, pc, translation_range=0.2):
        translation = np.random.uniform(-translation_range, translation_range, size=(1, 3))
        return pc + translation

    def random_jitter(self, pc, sigma=0.01, clip=0.05):
        noise = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
        return pc + noise


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        pc = sample['point_cloud']  # (N, 3)
        seg = sample['seg_label']   # (N,)
        center = sample['box_center']
        angle = sample['box_angle']
        box_dims = sample['box_dims']
        obj_class = sample['class']

        # Data augmentation
        pc = self.random_rotation(pc)
        pc = self.random_scaling(pc)
        pc = self.random_translation(pc)
        pc = self.random_jitter(pc)

        # Randomly sample npoints
        if pc.shape[0] >= self.npoints:
            choice = np.random.choice(pc.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(pc.shape[0], self.npoints, replace=True)
        pc_sampled = pc[choice, :]
        seg_sampled = seg[choice]

        return {
            'point_cloud': torch.from_numpy(pc_sampled),        # (npoints, 3)
            'seg_label': torch.from_numpy(seg_sampled),         # (npoints,)
            'center_label': torch.from_numpy(center),           # (3,)
            'angle_label': torch.tensor(angle, dtype=torch.float32), # scalar
            'size_label': torch.from_numpy(box_dims),           # (3,)
            'class': obj_class
        }

