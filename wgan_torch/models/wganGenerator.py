import sys, os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class wganGenerator(nn.Module):
    def __init__(self):
        super(wganGenerator, self).__init__()


