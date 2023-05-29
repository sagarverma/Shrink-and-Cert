import torch
import torch.nn.functional as F 

import numpy as np
from scipy.stats import truncnorm 


def l2_normalize(x):
    return x / (torch.sqrt(torch.sum(x**2.)) + 1e-9)

def trunc(shape):
    return torch.from_numpy(truncnorm.rvs(0.5, 1, size=shape)).float()

def linear_lipschitz(w, power_iters=5):
    rand_x = trunc(w.shape[1]).type_as(w)
    for _ in range(power_iters):
        x = l2_normalize(rand_x)
        x_p = F.linear(x, w) 
        rand_x = F.linear(x_p, w.T)

    lc = torch.sqrt(torch.abs(torch.sum(w @ x)) / (torch.abs(torch.sum(x)) + 1e-9)).data.cpu().item()
    return lc

def conv_lipschitz(w, in_channels, stride=1, padding=0, power_iters=5):
    rand_x = trunc((1, in_channels, 32, 32)).type_as(w)
    for _ in range(power_iters):
        x = l2_normalize(rand_x)
        x_p = F.conv2d(x, w, 
                       stride=stride, 
                       padding=padding) 
        rand_x = F.conv_transpose2d(x_p, w, 
                                    stride=stride, 
                                    padding=padding)

    Wx = F.conv2d(rand_x, w, 
                  stride=stride, padding=padding)
    lc = torch.sqrt(torch.abs(torch.sum(Wx**2.)) / 
                    (torch.abs(torch.sum(rand_x**2.)) + 1e-9)).data.cpu().item()
    return lc

def get_lipschitz(weight):
    lip = {}
    for k, v in weight.items():
        if "popup" in k:
            if 'conv' in k or 'linear' in k or 'features' in k or 'classifier' in k:
                w = weight[k.replace('popup_scores', 'weight')]
                m = torch.clamp(weight[k], 0, 0.15)
                final = w * m
                if 'conv' in k or 'features' in k:
                    lc = conv_lipschitz(final, v.shape[1], power_iters=10)
                    lc_m = conv_lipschitz(m, v.shape[1], power_iters=10)
                    lc_org = conv_lipschitz(w, v.shape[1], power_iters=10)
                else:
                    lc = linear_lipschitz(final)
                    lc_m = linear_lipschitz(m)
                    lc_org = linear_lipschitz(w)

                lip[k] = {"lc": lc, "lc_m": lc_m , "lc_org": lc_org}

    return lip