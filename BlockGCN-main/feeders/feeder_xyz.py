import torch
import numpy as np
from torch.utils.data import Dataset
from . import tools

# COCO keypoint pairs for bone calculation
coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), 
              (10, 8), (11, 9), (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]

class Feeder(Dataset):
    def __init__(self, data_path: str, data_split: str, p_interval: list=[0.95], window_size: int=64, bone: bool=False, vel: bool=False):
        """
        Initialize Feeder class with parameters for loading and preprocessing data.

        :param data_path: Path to the dataset (NPZ file)
        :param data_split: Data split ('train' or 'test')
        :param p_interval: List of probabilities for cropping the sequence
        :param window_size: The size of the output window
        :param bone: Whether to use bone modality (relative joint coordinates)
        :param vel: Whether to use velocity modality (differences between consecutive frames)
        """
        super().__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.load_data()

    def load_data(self):
        """Load data from the NPZ file based on the specified data split."""
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.data_split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train']
        elif self.data_split == 'test':
            self.data = npz_data['x_test']
            self.label = npz_data['y_test']
        else:
            raise ValueError(f"Invalid data split '{self.data_split}'")

        self.sample_name = [f"{self.data_split}_{i}" for i in range(len(self.data))]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Retrieve a single sample (data, label) and apply necessary transformations.

        :param idx: Index of the sample
        :return: Transformed data sample, corresponding label, and index
        """
        data_sample = np.array(self.data[idx])  # M T V C
        label = self.label[idx]
        valid_frame_num = np.sum(data_sample.sum(0).sum(-1).sum(-1) != 0)

        if valid_frame_num == 0:
            return np.zeros((3, 64, 17, 2)), label, idx  # Return zero array if no valid frames

        # Resize and preprocess the data
        data_sample = tools.valid_crop_resize(data_sample, valid_frame_num, self.p_interval, self.window_size)

        # Apply bone transformation if enabled
        if self.bone:
            data_sample = self.compute_bone_data(data_sample)

        # Apply velocity transformation if enabled
        if self.vel:
            data_sample = self.compute_velocity(data_sample)

        # Normalize data by subtracting the first joint (spine center)
        data_sample -= np.tile(data_sample[:, :, 0:1, :], (1, 1, 17, 1))

        return data_sample, label, idx

    def compute_bone_data(self, data_sample):
        """Compute bone data from joint data."""
        bone_data = np.zeros_like(data_sample)
        for v1, v2 in coco_pairs:
            bone_data[:, :, v1 - 1] = data_sample[:, :, v1 - 1] - data_sample[:, :, v2 - 1]
        return bone_data

    def compute_velocity(self, data_sample):
        """Compute the velocity of the joints."""
        data_sample[:, :-1] = data_sample[:, 1:] - data_sample[:, :-1]
        data_sample[:, -1] = 0  # Set the last frame velocity to zero
        return data_sample

    def top_k(self, score, top_k):
        """
        Calculate top-K accuracy.

        :param score: The model's predicted scores
        :param top_k: The top K value
        :return: The top-K accuracy
        """
        rank = score.argsort()
        hit_top_k = [label in rank[i, -top_k:] for i, label in enumerate(self.label)]
        return sum(hit_top_k) / len(hit_top_k)
