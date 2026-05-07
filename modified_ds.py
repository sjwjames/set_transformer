import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import h5py
import pdb
from tqdm import tqdm, trange

from models import DeepSet


class PermEqui1_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui1_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui2_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    xm = self.Lambda(xm)
    x = self.Gamma(x)
    x = x - xm
    return x

class PermEqui2_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    xm = self.Lambda(xm)
    x = self.Gamma(x)
    x = x - xm
    return x


class DTanh(nn.Module):

    def __init__(self, x_dim, d_dim, output_dim,num_outputs, pool='mean'):
        super(DTanh, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs
        self.pool = pool
        if pool == 'max':
            self.phi = nn.Sequential(
                PermEqui2_max(self.x_dim, self.d_dim),
                nn.ELU(),
                PermEqui2_max(self.d_dim, self.d_dim),
                nn.ELU(),
                PermEqui2_max(self.d_dim, self.d_dim),
                nn.ELU(),
            )
        elif pool == 'max1':
            self.phi = nn.Sequential(
                PermEqui1_max(self.x_dim, self.d_dim),
                nn.ELU(),
                PermEqui1_max(self.d_dim, self.d_dim),
                nn.ELU(),
                PermEqui1_max(self.d_dim, self.d_dim),
                nn.ELU(),
            )
        elif pool == 'mean':
            self.phi = nn.Sequential(
                PermEqui2_mean(self.x_dim, self.d_dim),
                nn.ELU(),
                PermEqui2_mean(self.d_dim, self.d_dim),
                nn.ELU(),
                PermEqui2_mean(self.d_dim, self.d_dim),
                nn.ELU(),
            )
        elif pool == 'mean1':
            self.phi = nn.Sequential(
                PermEqui1_mean(self.x_dim, self.d_dim),
                nn.ELU(),
                PermEqui1_mean(self.d_dim, self.d_dim),
                nn.ELU(),
                PermEqui1_mean(self.d_dim, self.d_dim),
                nn.ELU(),
            )

        self.projection = nn.Sequential(
            # nn.Linear(self.d_dim, self.d_dim),
            # nn.ELU(),
            nn.Linear(self.d_dim, self.d_dim),
            # nn.LayerNorm(self.d_dim),
            nn.ReLU(),
            nn.Linear(self.d_dim, self.d_dim),
            # nn.LayerNorm(self.d_dim),
            nn.ReLU(),
            nn.Linear(self.d_dim, self.d_dim),
            # nn.LayerNorm(self.d_dim),
            nn.ReLU(),
            # nn.Linear(self.d_dim, self.output_dim*self.num_outputs),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(self.d_dim, self.d_dim),
            nn.ReLU(),
            nn.Linear(self.d_dim, self.output_dim * self.num_outputs)
        )

    def forward(self, x):
        phi_output = self.phi(x)
        projected_elements = self.projection(phi_output)
        if self.pool in ["max", "max1"]:
            pooled_output, _ = projected_elements.max(1)
        else:
            pooled_output = projected_elements.mean(1)
        pooled_output = self.last_layer(pooled_output)
        return pooled_output.reshape(-1, self.num_outputs, self.output_dim)


def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm