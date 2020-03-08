import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data.dataGrabber


use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

DIM = 64
BATCH_SIZE = 50
CRITIC_ITERS = 5
LAMBDA = 10
ITERS = 200000
OUTPUT_DIM = 6


class wganModel():
    '''
        Hier kommt noch was Schlaues hin...
    '''
    def __init__(self, DB_FILE, DB_NAME):
        self.dG = data.dataGrabber(DB_FILE, DB_NAME)
        self.ALL_EVENTS = np.array([self.dG.reNbRows('augeralle'), self.dG.reNbColumns('augeralle')])