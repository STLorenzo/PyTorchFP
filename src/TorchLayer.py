import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
import time  # Time measuring library
import datetime
import signal
# Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.rmsprop import RMSprop
# Personal Libraries
from src.general_functions import *


class TorchLayer(nn.Module):
    def __init__(self, constructor, params_dict):
        super().__init__()
        self.constructor = constructor
        self.params_dict = params_dict
        self.functions_wp = []
        self.layer = self.construct()

    def construct(self):
        return self.constructor(**self.params_dict)

    def add_function_with_parameters(self, constructor, params=None):
        if params is None:
            params = {}
        self.functions_wp.append((constructor, params))

    def forward_data(self, x):
        x = self.layer(x)
        for f_wp in self.functions_wp:
            function = f_wp[0]
            params = f_wp[1]
            x = function(x, **params)
        return x
