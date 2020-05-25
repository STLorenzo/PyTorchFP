import cv2  # library of computer-vision functions
import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
# Torch Libraries
import torch
# Personal libraries
from src.general_functions import *

class DataLoader:
    conf_filename = None
    project_conf_filename = None

    def make_data(self, val_pct=None):
        pass

    def save_Xy(self, train_X, train_y, test_X, test_y):
        pass

    def read_XY(self):
        pass