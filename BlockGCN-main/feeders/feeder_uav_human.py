import torch
import numpy as np
from torch.utils.data import Dataset
from .feeder_xyz import Feeder
from . import tools
from .bone_pairs import coco_pairs

class FeederUAVHuman(Feeder):
    def __init__(self, data_path, label_path=None, p_interval=1, data_split='train', random_choose=False,
                 random_shift=False, random_move=False, random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False):
        """
        Initialize FeederUAVHuman object with various parameters.

        :param data_path: Path to the input data
        :param label_path: Path to the label data
        :param p_interval: Interval for sampling data
        :param data_split: 'train' or 'test'
        :param random_choose: If True, randomly select a portion of the input sequence
        :param random_shift: If True, apply random shift to the data
        :param random_move: If True, apply random movements
        :param random_rot: If True, apply random rotation to the skeleton data
        :param window_size: The length of the output sequence
        :param normalization: If True, normalize the input data
        :param debug: If True, only load the first 100 samples for debugging
        :param use_mmap: If True, use memory-mapped file loading
        :param bone: Whether to use bone modality
        :param vel: Whether to use velocity modality
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.data_split = data_split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.bone = bone
        self.vel = vel
        self.load_data()

        if normalization:
            self.compute_mean_std()

    def load_data(self):
        data = np.load(self.data_path)
        label = np.load(self.label_path)

        data_type_name = 'test_' if self.data_split == 'test' else 'train_'
        sample_limit = 100 if self.debug else len(data)

        self.data = data[:sample_limit]
        self.label = label[:sample_limit]
        self.sample_name = [f"{data_type_name}{i}" for i in range(sample_limit)]

    def compute_mean_std(self):

        N, C, T, V, M = self.data.shape
        self.mean_map = self.data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = self.data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_sample = self.data[index]
        label = self.label[index]
        
        valid_frame_num = np.sum(data_sample.sum(0).sum(-1).sum(-1) != 0)
        
        if valid_frame_num == 0:
            zero_tensor = np.zeros((3, 64, 17, 2))
            return torch.from_numpy(zero_tensor), torch.from_numpy(zero_tensor), label, index

        data_sample = tools.valid_crop_resize(data_sample, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot:
            data_sample = tools.random_rot(data_sample)

        if self.bone:
            bone_data = self._compute_bone_data(data_sample)
            data_sample = bone_data

        if self.vel:
            data_sample[:, :-1] = data_sample[:, 1:] - data_sample[:, :-1]
            data_sample[:, -1] = 0

        joint_data = self._normalize_joint(data_sample)
        return joint_data, data_sample, label, index

    def _compute_bone_data(self, data_sample):
        bone_data = np.zeros_like(data_sample)
        for v1, v2 in coco_pairs:
            bone_data[:, :, v1 - 1] = data_sample[:, :, v1 - 1] - data_sample[:, :, v2 - 1]

        center_index = 8
        bone_data[:, :, center_index] = data_sample[:, :, center_index]
        return bone_data

    def _normalize_joint(self, data_sample):
        center_index = 8
        trajectory = data_sample[:, :, center_index]
        data_sample -= data_sample[:, :, center_index:center_index + 1]
        data_sample[:, :, center_index] = trajectory
        return data_sample

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [label in rank[i, -top_k:] for i, label in enumerate(self.label)]
        return sum(hit_top_k) / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
