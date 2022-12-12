# 此文件定义spike neuron network中常用的loss
import torch.nn as nn
from model.surrogate_act import *
import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


class STCA_ClassifyLoss(nn.Module):
    """
        v_1: 使用循环保证TCA分类损失函数正确
    """

    def __init__(self, cfg):
        super(STCA_ClassifyLoss, self).__init__()
        self.cfg = cfg
        print('Use Classification Loss in TCA ')

    def forward(self, vmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        loss = torch.tensor(0.0, device=self.cfg.device)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= self.thresh)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > self.cfg.C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0: num_cluster = 0
                end_list = torch.tensor(end_list, device=self.cfg.device)
                beg_list = torch.tensor(beg_list, device=self.cfg.device)
                # 没有脉冲簇而label为1
                if labels[ibatch, ineuron] > (num_cluster > 0):
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - self.cfg.C, 0)
                        rmax = min(cluster_pos_spike[-1] + self.cfg.C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if (torch.sum(mask == 0) <= 0):
                        loss += v[pos_spike[random.randint(0, pos_spike.numel() - 1)]] - self.cfg.thresh
                    else:
                        loss += self.cfg.thresh - torch.max(v[mask == 0])
                # 有脉冲簇而label为0
                if labels[ibatch, ineuron] < (num_cluster > 0):
                    idx_cluster = torch.argmin(end_list - beg_list)
                    loss += v[pos_spike[end_list[idx_cluster]]] - self.cfg.thresh
        return loss


class STCA_TCA_Loss(nn.Module):
    def __init__(self):
        super(STCA_TCA_Loss, self).__init__()
        print('Use TCA Loss')

    def forward(self, vmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        loss = torch.tensor(0.0, device=self.cfg.device)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= self.cfg.thresh)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > self.cfg.C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0: num_cluster = 0
                end_list = torch.tensor(end_list, device=self.cfg.device)
                beg_list = torch.tensor(beg_list, device=self.cfg.device)
                # 发放脉冲簇过少
                if labels[ibatch, ineuron] > num_cluster:
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - self.cfg.C, 0)
                        rmax = min(cluster_pos_spike[-1] + self.cfg.C, num_time - 1)
                        mask[lmin:rmax + 1] = 1
                    if (torch.sum(mask == 0) <= 0):
                        loss += v[pos_spike[random.randint(0, pos_spike.numel() - 1)]] - self.cfg.thresh
                    else:
                        loss += self.cfg.thresh - torch.max(v[mask == 0])
                # 发放脉冲簇过多
                if labels[ibatch, ineuron] < num_cluster:
                    idx_cluster = torch.argmin(end_list - beg_list)
                    loss += v[pos_spike[end_list[idx_cluster]]] - self.cfg.thresh
        return loss
