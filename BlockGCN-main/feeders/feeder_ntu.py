import numpy as np
from torch.utils.data import Dataset
from feeders import tools
from .bone_pairs import ntu_pairs

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False,
                 random_shift=False, random_move=False, random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False):
        """
        Initialize Feeder class with parameters for data preprocessing and transformations.

        :param data_path: Path to the data file (NPZ format)
        :param label_path: Path to the label data (optional)
        :param split: 'train' or 'test'
        :param random_choose: Whether to randomly select a portion of the input sequence
        :param random_shift: Whether to randomly shift the sequence
        :param random_move: Whether to apply random movements
        :param random_rot: Whether to apply random rotation to the data
        :param window_size: Output sequence length
        :param normalization: Whether to normalize the data
        :param debug: If True, only use the first 100 samples for debugging
        :param use_mmap: Whether to use memory-mapped files for loading data
        :param bone: Whether to use bone modality
        :param vel: Whether to use velocity modality
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
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
        """Load data and labels from the NPZ file based on the specified split."""
        npz_data = np.load(self.data_path)
        x_key = f'x_{self.split}'  # Dynamically choose the correct data key
        y_key = f'y_{self.split}'

        if x_key not in npz_data or y_key not in npz_data:
            raise ValueError(f"Data split '{self.split}' not found in {self.data_path}")

        self.data = npz_data[x_key]
        self.label = np.where(npz_data[y_key] > 0)[1]
        self.sample_name = [f"{self.split}_{i}" for i in range(len(self.data))]

        # Reshape data to (N, C, T, V, M) format
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def compute_mean_std(self):
        """Compute the mean and standard deviation for data normalization."""
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.label)

    def __iter__(self):
        """Return the iterator object."""
        return self

    def __getitem__(self, index):
        """Retrieve a data sample, process it, and return it with the label."""
        data_sample = self.data[index]
        label = self.label[index]

        # Determine number of valid frames
        valid_frame_num = np.sum(data_sample.sum(0).sum(-1).sum(-1) != 0)

        # If no valid frames, return zero tensors
        if valid_frame_num == 0:
            zero_tensor = np.zeros((3, 64, 17, 2))
            return torch.from_numpy(zero_tensor), torch.from_numpy(zero_tensor), label, index

        # Preprocess and resize the data
        data_sample = tools.valid_crop_resize(data_sample, valid_frame_num, self.p_interval, self.window_size)

        # Apply transformations
        joint = self.apply_random_rotation(data_sample)
        data_sample = self.apply_bone_modality(data_sample) if self.bone else self.apply_joint_normalization(data_sample)

        if self.vel:
            data_sample = self.compute_velocity(data_sample)

        return joint, data_sample, label, index

    def apply_random_rotation(self, data_sample):
        """Apply random rotation to the data if specified."""
        if self.random_rot:
            data_sample = tools.random_rot(data_sample)
        return data_sample

    def apply_bone_modality(self, data_sample):
        """Apply bone modality transformation."""
        bone_data = np.zeros_like(data_sample)
        for v1, v2 in ntu_pairs:
            bone_data[:, :, v1 - 1] = data_sample[:, :, v1 - 1] - data_sample[:, :, v2 - 1]

        # Preserve the spine center's trajectory
        bone_data[:, :, 20] = data_sample[:, :, 20]
        return bone_data

    def apply_joint_normalization(self, data_sample):
        """Normalize the joint data to the spine center."""
        trajectory = data_sample[:, :, 20]
        data_sample -= data_sample[:, :, 20:21]
        data_sample[:, :, 20] = trajectory
        return data_sample

    def compute_velocity(self, data_sample):
        """Compute velocity of the data."""
        data_sample[:, :-1] = data_sample[:, 1:] - data_sample[:, :-1]
        data_sample[:, -1] = 0
        return data_sample

    def top_k(self, score, top_k):
        """Calculate top-K accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) / len(hit_top_k)


def import_class(name):
    """Dynamically import a class from a module given its full path."""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
