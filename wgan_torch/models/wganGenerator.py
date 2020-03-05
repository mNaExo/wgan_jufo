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
        self.MAIN_MODULE = nn.Sequential(
            nn.Linear(in_features=2, out_features=3),
            nn.ReLU(),

            nn.Linear(in_features=3, out_features=4),
            nn.ReLU(),

            nn.Conv2d(in_channels=4, out_channels=4),
            nn.ReLU(),

            nn.Linear(in_features=4, out_features=6),
            nn.ReLU()
        )
        self.OUTPUT = nn.Tanh()

    def forward(self, x):
        x = self.MAIN_MODULE(x)
        return self.OUTPUT(x)