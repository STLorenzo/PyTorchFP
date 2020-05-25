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


class TorchNet(nn.Module):
    def __init__(self, data_loader, device=None, loss_function_name=None, optimizer_name=None, lr=None):
        super().__init__()
        self.STOP_TRAIN = False
        self.loss_dict = None
        self.optimizer_dict = None
        self.conf_filename = None
        self.project_conf_filename = None
        self.p_conf_data = None
        self.l_conf_data = None

        self.device = None
        self.lr = None
        self.optimizer = None
        self.loss_function = None

        self.MAX_VAL_TRAIN_PCT = None
        self.val_train_pct = None

        self.base_path = None
        self.data_base_path = None
        self.created_data_path = None
        self.models_path = None
        self.half_trained_model_path = None
        self.logs_path = None

        self.prepare_signal_handlers()

        self.establish_arquitecture()
        self.compile_net()

    def assign_conf_data(self):
        if self.project_conf_filename is None or self.conf_filename is None:
            raise Exception("Filenames not defined")

        self.p_conf_data = read_conf(self.project_conf_filename)
        self.l_conf_data = read_conf(self.conf_filename)

    def assign_paths(self):
        if self.p_conf_data is None or self.l_conf_data is None:
            raise Exception("Conf data not defined")

        self.base_path = Path(self.p_conf_data['base_path'])
        self.data_base_path = self.base_path / self.p_conf_data['dirs']['data_dir']
        self.created_data_path = self.data_base_path / self.l_conf_data['dirs']['created_data_dir']
        self.models_path = self.created_data_path / self.l_conf_data['dirs']['models_dir']
        self.half_trained_model_path = self.created_data_path / self.l_conf_data['dirs']['half_trained_models_dir']
        self.logs_path = self.created_data_path / self.l_conf_data['dirs']['logs_dir']

        create_dir(self.created_data_path)
        create_dir(self.models_path)
        create_dir(self.half_trained_model_path)
        create_dir(self.logs_path)

    def prepare_signal_handlers(self):
        signal.signal(signal.SIGINT, self.catch_abrupt_end)
        signal.signal(signal.SIGTERM, self.catch_abrupt_end)

    def catch_abrupt_end(self, signum, frame):
        self.STOP_TRAIN = True

    def establish_arquitecture(self):
        pass

    def compile_net(self):
        pass

    def forward(self, x):
        pass

    def fwd_pass(self, X, y, loss_function=None, optimizer=None, train=False):
        pass

    def train_p(self, train_X=None, train_y=None, batch_size=None, epoch=0, max_epochs=None, log_file=None,
                loss_function=None, val_train_pct=None,
                optimizer=None, model_name=f"model-{time.time()}", n_steps_log=None, verbose=False):
        pass

    def train_test(self, batch_X, batch_y, test_X, test_y, loss_function, optimizer, val_train_pct,
                   f, model_name, epoch, loss_function_name, optimizer_name, lr, batch_size, acc, loss):
        pass

    def test_p(self, test_X=None, test_y=None, loss_function=None, optimizer=None, size=None, verbose=False):
        pass

    def make_predictions(self):
        pass

    def check_optim_loss(self, loss_function, optimizer):
        if (optimizer is None and self.optimizer is None) or (loss_function is None and self.loss_function is None):
            raise Exception("Net not compiled")

        if optimizer is None:
            optimizer = self.optimizer
        if loss_function is None:
            loss_function = self.loss_function

        return loss_function, optimizer

    def print_instance(self, epoch, max_epoch, batch_size, optimizer, loss_function):
        print(f"MODEL: {self.state_dict()}\n"
              f"OPTIMIZER: {optimizer.state_dict()}\n"
              f"Epoch: {epoch}\n"
              f"Max Epoch: {max_epoch}\n"
              f"Batch_size: {batch_size}\n"
              f"Loss: {self.get_loss_function_name(loss_function)}\n")

    def save_instance_net(self, path, epoch, max_epoch, batch_size, optimizer, loss_function,
                          log_file, model_name):
        torch.save({
            'epoch': epoch,
            'max_epoch': max_epoch,
            'batch_size': batch_size,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': self.get_loss_function_name(loss_function),
            'log_file': log_file,
            'model_name': model_name,
        }, path)

    def load_instance_net(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        max_epoch = checkpoint['max_epoch']
        loss_function = self.get_loss_function_by_name(checkpoint['loss'])()
        batch_size = checkpoint['batch_size']
        log_file = checkpoint['log_file']
        model_name = checkpoint['model_name']
        return epoch, max_epoch, loss_function, batch_size, log_file, model_name

    def save_net(self, filename=None):
        if filename is None:
            filename = self.l_conf_data['net_save_default_name']
        path = self.models_path / filename
        torch.save(self.state_dict(), path)

    def load_net(self, filename=None):
        if filename is None:
            filename = self.l_conf_data['net_save_default_name']
        path = self.models_path / filename
        self.load_state_dict(torch.load(path))

    @staticmethod
    def get_optimizer_data(optimizer):
        s = str(optimizer)
        name = s.rsplit(' (')[0]
        lr = s.rsplit('lr: ', 1)[1].rsplit('\n')[0]
        return name, lr

    @staticmethod
    def get_loss_function_name(loss_function):
        return str(loss_function)[:-2]

    def get_loss_function_by_name(self, loss):
        if loss in self.loss_dict.keys():
            return self.loss_dict[loss]
        else:
            raise Exception(f"Loss name not doesn't match available functions\n"
                            f"{loss} - {self.loss_dict}")

    def get_optimizer_by_name(self, optimizer_name):
        if optimizer_name in self.optimizer_dict.keys():
            return self.optimizer_dict[optimizer_name]
        else:
            raise Exception(f"Optimizer name not doesn't match available optimizers\n"
                            f"{optimizer_name} - {self.optimizer_dict.keys()}")


def optimize(device, img_loader, loss_functions_names=None, optimizers_names=None, batch_sizes=None, lrs=None,
             epochs=None,
             log_file=None):
    pass
