import random
import numpy as np
import torch
import torch.nn.functional as F


# Helper functions
def get_valid_frame_range(data_numpy):
    
    valid_frame = (data_numpy != 0).sum(axis=(2, 3)).sum(axis=0) > 0
    start = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    return start, end


def generate_random_crop_size(valid_size, p_interval):
   
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        return bias, valid_size - 2 * bias
    else:
        p = np.random.uniform(p_interval[0], p_interval[1])
        crop_size = np.clip(int(np.floor(valid_size * p)), 64, valid_size)
        bias = np.random.randint(0, valid_size - crop_size + 1)
        return bias, crop_size


# Data preprocessing functions
def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    C, T, V, M = data_numpy.shape
    start, end = get_valid_frame_range(data_numpy)
    valid_size = end - start

    # Crop
    bias, cropped_length = generate_random_crop_size(valid_size, p_interval)
    cropped_data = data_numpy[:, start + bias:start + bias + cropped_length, :, :]

    # Resize
    cropped_data = torch.tensor(cropped_data, dtype=torch.float)
    cropped_data = cropped_data.permute(0, 2, 3, 1).view(C * V * M, cropped_length)
    cropped_data = cropped_data[None, None, :, :]
    resized_data = F.interpolate(cropped_data, size=(C * V * M, window), mode='bilinear', align_corners=False).squeeze()
    resized_data = resized_data.view(C, V, M, window).permute(0, 3, 1, 2).numpy()

    return resized_data


def downsample(data_numpy, step, random_sample=True):
    
    start = np.random.randint(step) if random_sample else 0
    return data_numpy[:, start::step, :, :]


def temporal_slice(data_numpy, step):
    
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T // step, step, V, M).transpose(0, 1, 3, 2, 4).reshape(C, T // step, V, step * M)


def mean_subtractor(data_numpy, mean):
    
    if mean == 0:
        return data_numpy
    C, T, V, M = data_numpy.shape
    start, end = get_valid_frame_range(data_numpy)
    data_numpy[:, :end, :, :] -= mean
    return data_numpy


def auto_padding(data_numpy, size, random_pad=False):
    
    C, T, V, M = data_numpy.shape
    if T < size:
        padding_start = random.randint(0, size - T) if random_pad else 0
        padded_data = np.zeros((C, size, V, M))
        padded_data[:, padding_start:padding_start + T, :, :] = data_numpy
        return padded_data
    return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        return auto_padding(data_numpy, size) if auto_pad else data_numpy
    else:
        start = random.randint(0, T - size)
        return data_numpy[:, start:start + size, :, :]


def random_move(data_numpy, angle_candidate=[-10., -5., 0., 5., 10.], scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2], move_time_candidate=[1]):
  
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    # Generate random transformation parameters
    angles = np.random.choice(angle_candidate, num_node)
    scales = np.random.choice(scale_candidate, num_node)
    t_x = np.random.choice(transform_candidate, num_node)
    t_y = np.random.choice(transform_candidate, num_node)

    # Interpolation for smooth transitions
    a, s, tx, ty = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(angles[i], angles[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(scales[i], scales[i + 1], node[i + 1] - node[i])
        tx[node[i]:node[i + 1]] = np.linspace(t_x[i], t_x[i + 1], node[i + 1] - node[i])
        ty[node[i]:node[i + 1]] = np.linspace(t_y[i], t_y[i + 1], node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]])

    # Apply transformations
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += tx[i_frame]
        new_xy[1] += ty[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros_like(data_numpy)
    start, end = get_valid_frame_range(data_numpy)
    size = end - start
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, start:end, :, :]
    return data_shift


def _rot(rot):
    
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = torch.zeros(rot.shape[0], 1)
    ones = torch.ones(rot.shape[0], 1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=1)

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    return rz.matmul(ry).matmul(rx)


def random_rot(data_numpy, theta=0.3):
    
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).view(T, C, V * M)
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rotation_matrix = _rot(rot)
    data_torch = torch.matmul(rotation_matrix, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3)
    return data_torch


def openpose_match(data_numpy):
    
    C, T, V, M = data_numpy.shape
    assert C == 3
    score = data_numpy[2, :, :, :].sum(axis=1)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # Data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # Data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # Matching pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward

    # Generate data based on forward map
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # Sort by score
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    return data_numpy[:, :, :, rank]
