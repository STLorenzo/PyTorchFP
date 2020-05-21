import os  # OS library
import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
import time  # Time measuring library
import datetime
import sys
import signal
# Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Personal Libraries
from general_functions import create_dir


# TODO: check signal handler
def siging_handler(sig, frame):
    print("Sigint received: Saving instance of training...")
    sys.exit(0)


class ImgConvNet(nn.Module):
    def __init__(self, img_loader, device=torch.device('cpu'), optimizer=None, loss_function=None, lr=1e-3):
        super().__init__()
        self.img_loader = img_loader
        self.lr = lr
        self.STOP_TRAIN = False
        self.half_trained_model_path = Path("../data/created_data")
        self.loss_dict = {'MSELoss': nn.MSELoss}

        signal.signal(signal.SIGINT, self.catch_sigint)

        # Conv2d(in_channels = 1, out_channels=32, kernel(window) = 5)
        # By default stride = 1, padding = 0
        # if kernel is a single int it creates a (5, 5) convolving kernel

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        self._flatten_dim = self.calculate_flatten_dim()

        # view -1 adapts to all samples size, 1 means the channels( I think)
        # Create random data adn run only through conv part to calculate the flattened dimension

        self.fc1 = nn.Linear(self._flatten_dim, 512)
        self.fc2 = nn.Linear(512, 2)

        # NET COMPILE
        self.device = device
        self.to(device)
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), self.lr)
        else:
            self.optimizer = optimizer
        if loss_function is None:
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = loss_function
        print(f"Net will run in {self.device}")

    def convs(self, x, verbose=False):
        # It scales down the data
        # Same effect using stride in conv layer except conv layer learns pool doesnt
        # max_pool2d( kernel = , stride=(2,2))
        # stride means how much the window moves
        # relu doesnt change shapes(sizes)
        if verbose:
            print("Before: ", x[0].shape)
            # Sizes are (1, img_h, img_w)
            x = self.conv1(x)
            # Sizes are (conv_output, img_h - kernel_w - 1, img_w - kernel_h - 1)
            # Notation -> (conv_output, new_size_w, new_size_h)
            print("After conv1: ", x[0].shape)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            # Sizes are ( conv_output, new_size_w/2, new_size_h/2)
            print("After max pool2d(2,2): ", x[0].shape)

            x = self.conv2(x)
            print("After conv2: ", x[0].shape)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            print("After max pool2d(2,2): ", x[0].shape)

            x = self.conv3(x)
            print("After conv3: ", x[0].shape)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            print("After max pool2d(2,2): ", x[0].shape)

        else:
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        return x

    def calculate_flatten_dim(self):
        x = torch.rand(self.img_loader.img_size[0], self.img_loader.img_size[1]).view(-1, 1,
                                                                                      self.img_loader.img_size[0],
                                                                                      self.img_loader.img_size[1])
        x = self.convs(x, False)
        dim = 1
        for n in x[0].shape:
            dim *= n
        return dim

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)  # Activation Function

    def check_optim_loss(self, loss_function, optimizer):
        if (optimizer is None and self.optimizer is None) or (loss_function is None and self.loss_function is None):
            raise Exception("Net not compiled")

        if optimizer is None:
            optimizer = self.optimizer
        if loss_function is None:
            loss_function = self.loss_function

        return loss_function, optimizer

    def fwd_pass(self, X, y, loss_function=None, optimizer=None, train=False):
        loss_function, optimizer = self.check_optim_loss(loss_function, optimizer)

        if train:
            self.zero_grad()
        outputs = self(X)  # Because inherits Net we call the class itself
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
        acc = matches.count(True) / len(matches)
        loss = loss_function(outputs, y)

        if train:
            loss.backward()
            optimizer.step()
        return acc, loss

    def train_p(self, train_X=None, train_y=None, batch_size=100, epoch=0, max_epochs=10, log_file=None, loss_function=None,
                optimizer=None, model_name=f"model-{time.time()}", n_steps_log=50, verbose=False):

        if train_X is None:
            train_X = self.img_loader.read_train_X()
        if train_y is None:
            train_y = self.img_loader.read_train_y()

        # TODO: CHECK TESTING
        # test_X = img_loader.read_test_X()
        # test_y = img_loader.read_test_y()

        if log_file is None:
            create_dir("../doc")
            log_file = Path(f"../doc/{datetime.datetime.now()}.log")

        loss_function, optimizer = self.check_optim_loss(loss_function, optimizer)

        optimizer_name, lr = self.get_optimizer_data(optimizer)
        loss_function_name = self.get_loss_function_name(loss_function)

        self.STOP_TRAIN = False

        if verbose:
            print(f"Starting Training of {model_name},{max_epochs},"
                  f"{loss_function_name},{optimizer_name},{lr},{batch_size}")
        t0 = time.time()
        with open(log_file, "a") as f:
            for epoch in range(epoch, max_epochs):
                # TODO: check stop train
                if self.STOP_TRAIN:
                    if verbose:
                        print("Stopping Training")
                    self.print_instance(epoch, max_epochs, batch_size, optimizer, loss_function)
                    self.save_instance_net(self.half_trained_model_path / f"__half__{model_name}.pt",
                                           epoch, max_epochs, batch_size, optimizer, loss_function,
                                           log_file, model_name)

                    sys.exit(0)
                if verbose:
                    print(f"Epoch: {epoch}/{max_epochs}")
                for i in tqdm(range(0, len(train_X), batch_size)):
                    batch_X = train_X[i:i + batch_size].view(-1, 1, self.img_loader.img_size[0],
                                                             self.img_loader.img_size[1]).to(self.device)
                    batch_y = train_y[i:i + batch_size].to(self.device)

                    acc, loss = self.fwd_pass(batch_X, batch_y, loss_function, optimizer, train=True)
                    if i % n_steps_log == 0:
                        val_acc, val_loss = self.test_p(batch_X, batch_y, loss_function, optimizer)
                        f.write(f"{model_name},{epoch},{round(time.time(), 3)},"
                                f"{loss_function_name},{optimizer_name},{lr},{batch_size},"
                                f"{round(float(acc), 2)},{round(float(loss), 4)},"
                                f"{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")

        t1 = time.time() - t0
        if verbose:
            print(f"Training Finished in {t1} seconds")

    def resume_training(self, path):
        epoch, max_epoch, loss_function, batch_size, log_file, model_name = self.load_instance_net(path)
        self.train_p(epoch=epoch, max_epochs=max_epoch, batch_size=batch_size, loss_function=loss_function,
                     optimizer=self.optimizer, log_file=model_name, model_name=model_name, verbose=True)

    def test_p(self, test_X=None, test_y=None, loss_function=None, optimizer=None, size=None, verbose=False):
        if test_X is None:
            test_X = self.img_loader.read_test_X()
        if test_y is None:
            test_y = self.img_loader.read_test_y()
        loss_function, optimizer = self.check_optim_loss(loss_function, optimizer)

        if size is None or size >= len(test_X):
            X = test_X
            y = test_y
        else:
            random_start = np.random.randint(len(test_X) - size)
            X, y = test_X[random_start:random_start + size], test_y[random_start:random_start + size]

        if verbose:
            print("Starting Testing")
        t0 = time.time()
        with torch.no_grad():
            val_acc, val_loss = self.fwd_pass(
                X.view(-1, 1, self.img_loader.img_size[0], self.img_loader.img_size[1]).to(self.device),
                y.to(self.device), loss_function, optimizer)
        t1 = time.time() - t0
        if verbose:
            print(f"Testing Finished in {t1} seconds")
        return val_acc, val_loss

    def make_predictions(self):
        predict_data_path = self.img_loader.predict_dir
        for file in os.listdir(predict_data_path):
            # try:
            img = self.img_loader.read_image(predict_data_path / file)
            img = torch.Tensor(img).view(-1, 1, self.img_loader.img_size[0], self.img_loader.img_size[1]).to(
                self.device)
            img = self.img_loader.normalize_img(img)
            output = torch.argmax(self(img))
            output = self.img_loader.classes[output]
            self.img_loader.show_image(predict_data_path / file, output)

            # except Exception as e:
            #     print(e)
            # print(f"{file} could not be loaded")

    def print_instance(self, epoch, max_epoch, batch_size, optimizer, loss_function):
        print(f"MODEL: {self.state_dict()}\n"
              f"OPTIMIZER: {optimizer.state_dict()}\n"
              f"Epoch: {epoch}\n"
              f"Max Epoch: {max_epoch}\n"
              f"Batch_size: {batch_size}\n"
              f"Loss: {self.get_loss_function_name(loss_function)}\n")

    # TODO: CHECK SAVE LOAD (in particular loss function)
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
        loss_function = self.get_loss_function_by_name(checkpoint['loss'])
        batch_size = checkpoint['batch_size']
        log_file = checkpoint['log_file']
        model_name = checkpoint['model_name']
        return epoch, max_epoch, loss_function, batch_size, log_file, model_name

    def save_net(self, path):
        torch.save(self.state_dict(), path)

    def load_net(self, path):
        self.load_state_dict(torch.load(path))

    def catch_sigint(self, signum, frame):
        self.STOP_TRAIN = True

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
            return self.loss_dict[loss]()
        else:
            raise Exception(f"Loss name not doesn't match available functions\n"
                            f"{loss} - {self.loss_dict}")

    def optimize(self, loss_functions=None, optimizers=None, batch_sizes=None, lrs=None, epochs=None):
        if lrs is None:
            lrs = [1e-3, 5e-3, 1e-2, 1e-4]
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 128]
        if epochs is None:
            epochs = [10, 30, 50]

        if loss_functions is None:
            loss_functions = [nn.MSELoss()]

        if optimizers is None:
            optimizers = []
            optimizers_constructors = [optim.Adam, optim.Adagrad, optim.SGD]
            for lr in lrs:
                for constructor in optimizers_constructors:
                    optimizers.append(constructor(self.parameters(), lr))

        i = 0
        for epoch in epochs:
            for batch_size in batch_sizes:
                for loss_function in loss_functions:
                    for optimizer in optimizers:
                        optim_name, lr = self.get_optimizer_data(optimizer)
                        loss_name = self.get_loss_function_name(loss_function)
                        self.train_p(batch_size=batch_size, epochs=epoch,
                                     loss_function=loss_function, optimizer=optimizer,
                                     model_name=f"{optim_name}_{lr}_{loss_name}_{epoch}_{batch_size}",
                                     log_file="../doc/optimizer.log", verbose=True)
                        i += 1

        print(f"Trained: {i} Different models")
