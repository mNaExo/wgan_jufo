import sys, os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class wganGenerator(nn.Module):
    '''
        Generatives neuronales Netzwerk. Konische Architektur,
        einmal Conv2D, kann man ja mal probieren...
    '''
    def __init__(self):
        super(wganGenerator, self).__init__()
        self.l1 = nn.Linear(in_features=2, out_features=3)

        self.l2 = nn.Linear(in_features=3, out_features=4)

        self.l3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2)

        self.l4 = nn.Linear(in_features=2*2, out_features=6)

        self.OUTPUT = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.max_pool2d(F.relu(self.l3(x), (2, 2)))
        x = F.relu(self.l4(x))
        return self.OUTPUT(x)