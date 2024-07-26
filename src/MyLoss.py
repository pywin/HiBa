""" Utilities """
import os
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable
from numpy import random
import math
from torch.nn.modules.loss import _Loss
import utils
from torch.distributions import MultivariateNormal as MVN

args = utils.get_args()
if torch.cuda.is_available():
    device = torch.device('cuda:' + args.GPU if torch.cuda.is_available() else 'cpu')  #

class BMCLossMD(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss, noise_var

def bmc_loss_md(pred, target, noise_var):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    I = torch.eye(pred.shape[-1]).to(device=device)
    #logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))
    logits = MVN(pred, noise_var*I).log_prob(target).view(-1, 1)
    #print(logits.size())
    #print(torch.arange(pred.shape[0]).size())
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).view(-1, 1).float().to(device=device))
    loss = loss * (2 * noise_var).detach()
    return loss

class P_loss2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt_lable, pre_lable):
        M, N, A, B = gt_lable.shape
        gt_lable = gt_lable - torch.mean(gt_lable, dim=3).view(M, N, A, 1)
        pre_lable = pre_lable - torch.mean(pre_lable, dim=3).view(M, N, A, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=3))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=3))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=3) / (aPow * bPow + 0.01)
        loss = 1 - torch.sum(torch.sum(torch.sum(pearson, dim=2), dim=1), dim=0)/(gt_lable.shape[0] * gt_lable.shape[1] * gt_lable.shape[2])
        return loss

class P_loss3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt_lable, pre_lable):
        M, N, A = gt_lable.shape
        gt_lable = gt_lable - torch.mean(gt_lable, dim=2).view(M, N, 1)
        pre_lable = pre_lable - torch.mean(pre_lable, dim=2).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=2))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=2))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=2) / (aPow * bPow + 0.001)
        loss = 1 - torch.sum(torch.sum(pearson, dim=1), dim=0)/(gt_lable.shape[0] * gt_lable.shape[1])
        return loss



class SP_loss(nn.Module):
    def __init__(self, device, clip_length=256, delta=3, loss_type=1, use_wave=False):
        super(SP_loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype = torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, pred = None, flag = None):  # all variable operation
        fps = 30

        hr = gt.clone()

        hr[hr.ge(self.high_bound)] = self.high_bound-1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t*tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t*tmp), dim=-1) ** 2

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound



        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).to(self.device) / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1;

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx



def get_loss(bvp_pre, hr_pre, bvp_gt, hr_gt, av, dataName, \
             loss_sig0, loss_l1, loss_dc, loss_bsrc, args, inter_num):
    k = 2.0 / (1.0 + np.exp(-10.0 * inter_num/args.max_iter)) - 1.0

    # k = 1.0
    # k1 K6 NP
    # k2 k4 SP
    # k3 k5 k7
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13 = args.k1, k*args.k2, args.k3, k*args.k4, args.k5, k*args.k6, k*args.k7, k*args.k8, k*args.k9, k*args.k10, k*args.k11, k*args.k12, k*args.k13

    weights, l_dc = loss_dc(av, hr_gt)
    if dataName == 'PURE':
        loss = (k1*loss_sig0(bvp_pre, bvp_gt) + k2*loss_bscr(av, hr_gt, weights)/10 + k9*loss_l1(torch.squeeze(hr_pre), hr_gt)/10)/2 + k2*loss_dc/10
    elif dataName == 'UBFC':
        loss = (k3 * loss_sig0(bvp_pre, bvp_gt) + k4 * loss_bscr(av, hr_gt, weights)/10 + k10*loss_l1(torch.squeeze(hr_pre), hr_gt)/10) /2 + k4*loss_dc/10
    elif dataName == 'BUAA':
        loss = (k5 * loss_sig0(bvp_pre, bvp_gt)+ k6 * loss_bscr(av, hr_gt, weights)/10/10 + k11*loss_l1(torch.squeeze(hr_pre), hr_gt)/10) / 2 + k5*loss_dc/10
    elif dataName == 'VIPL':
        loss = k7 * loss_bscr(av, hr_gt, weights)/10 + k12*loss_l1(torch.squeeze(hr_pre), hr_gt)/10 + k12*loss_dc/10
    elif dataName == 'V4V':
        loss = k8 * loss_bscr(av, hr_gt, weights)/10 + k13*loss_l1(torch.squeeze(hr_pre), hr_gt)/10 + k13*loss_dc/10

    if torch.sum(torch.isnan(loss)) > 0:
        print('Tere in nan loss found in' + dataName)
    return loss
