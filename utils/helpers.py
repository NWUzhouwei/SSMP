# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv1d or \
       type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose1d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_volume_views(volume):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        sampled_xyz: sampled pointcloud data, [B, 3, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    sampled_xyz = torch.zeros(B, C, npoint, dtype=xyz.dtype).to(device)
    distance = torch.ones(B, N, dtype=xyz.dtype).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        # 更新第i个最远点
        sampled_xyz[:, :, i] = xyz[batch_indices, farthest, :]
        # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # 计算点集中的所有点到这个最远点的欧式距离
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = torch.argmax(distance, dim=1)


    return sampled_xyz

def chamfer_distance_with_batch(p1, p2, type):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param type: sum or mean
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    assert type == 'sum' or type == 'mean'

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)

    p1 = p1.transpose(1, 2)

    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist = torch.norm(dist, 2, dim=3)

    dist = torch.min(dist, dim=2)[0]

    if type == 'mean':
        dist = torch.mean(dist)
    elif type == 'sum':
        dist = torch.sum(dist)

    return dist

