import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from core.encoders import *
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *


class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)

        self.l2 = nn.Linear(512, 1)
    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

