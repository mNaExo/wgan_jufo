import sys, os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class wganGenerator(nn.Module):
    def __init__(self):
        super(wganGenerator, self).__init__()
        self.inL = nn.Linear(2, 3)
        self.hid1 = nn.Linear(3, 4)
        self.hid2 = nn.Conv2d(4, 4)
        self.hid3 = nn.Linear(4, 6)
        self.outL = nn.Linear(6, 6)
