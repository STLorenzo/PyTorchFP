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
from src.TorchNet import TorchNet
from src.TorchLayer import TorchLayer


class ConvNet(TorchNet):
    """
    Class that implements a TorchNet to create convolution nets
    """

    def __init__(self, img_loader, device=None, loss_function_name=None, optimizer_name=None, lr=None):
        # Initializes all not given variables to None and the signal handlers
        super().__init__(img_loader, device, lr)
        # Defined Paths
        self.project_conf_path = '/config/Project_conf.json'
        self.conf_path = "/config/ImgConvNet_conf.json"
        # Defined dicts
        self.loss_dict = {'MSELoss': nn.MSELoss,
                          'CrossEntropyLoss': nn.CrossEntropyLoss}
        self.optimizer_dict = {'Adam': optim.Adam,
                               'AdamW': optim.AdamW,
                               'SGD': optim.SGD,
                               'rmsprop': RMSprop}
        # Define the conv and dense layers
        self.conv_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self._flatten_dim = None

        # Get default configuration data and assign it to the internal variables
        self.assign_conf_data()
        # Assigns values to all path internal variables
        self.assign_paths()

        self.MAX_VAL_TRAIN_PCT = self.l_conf_data['max_val_train_pct']
        self.val_train_pct = self.l_conf_data['val_train_pct']

        # Get default data if not given
        if self.lr is None:
            self.lr = self.l_conf_data['lr']
        if self.device is None:
            self.device = torch.device(self.l_conf_data['device'])

        # Assign the architecture
        self.establish_architecture()
        # Compile the net once all parameters have been satisfied
        self.compile_net(self.device, loss_function_name, optimizer_name)

    def establish_architecture(self):
        """
        Method that establishes the architecture of the net. Assigning all its layers and
        activation functions with each parameters.
        """
        # Conv2d(in_features, out_features, kernel)
        # if kernel is only a number n transforms it to (n, n)
        # Aditional features to store: activation function and Max_pooling with kernel
        self.conv_layers.append(TorchLayer(nn.Conv2d, {'in_channels': 1,
                                                       'out_channels': 32,
                                                       'kernel_size': 5}))
        self.conv_layers.append(TorchLayer(nn.Conv2d, {'in_channels': 32,
                                                       'out_channels': 64,
                                                       'kernel_size': 5}))
        self.conv_layers.append(TorchLayer(nn.Conv2d, {'in_channels': 64,
                                                       'out_channels': 128,
                                                       'kernel_size': 5}))

        for conv_layer in self.conv_layers:
            conv_layer.add_function_with_parameters(F.relu)
            conv_layer.add_function_with_parameters(F.max_pool2d, {'kernel_size': (2, 2)})

        self._flatten_dim = self.calculate_flatten_dim()

        # view -1 adapts to all samples size, 1 means the channels( I think)
        # Create random data adn run only through conv part to calculate the flattened dimension

        layer = TorchLayer(nn.Linear, {'in_features': self._flatten_dim, 'out_features': 512})
        layer.add_function_with_parameters(F.relu)
        self.dense_layers.append(layer)
        layer = TorchLayer(nn.Linear, {'in_features': 512, 'out_features': 2})
        layer.add_function_with_parameters(F.softmax, {'dim': 1})
        self.dense_layers.append(layer)

    def calculate_flatten_dim(self):
        """
        Method that calculates the dimension of the dense layer based in its previous convolution
        layers

        Returns
        -------
        dim : int
            dimension that the dense layer has to have
        """
        x = torch.rand(self.data_loader.get_image_size()).view(self.data_loader.get_input_size())
        x = self.forward_data_through_layers(self.conv_layers, x)
        dim = 1
        for n in x[0].shape:
            dim *= n
        return dim

    def forward_data_through_layers(self, layers, x):
        """
        Method that forwards the data through the layers

        Parameters
        ----------
        layers : list
            list of layers from which to forward the data
        x :
            the data to be forwarded

        Returns
        -------
        x :
            the data after being forwarded
        """
        for layer in layers:
            x = layer.forward_data(x)
        return x

    def compile_net(self, device, loss_function_name, optimizer_name):
        """
        Method that compiles the net establishing the device where is going to run (cpu or gpu),
        loss_function and optimizer name

        Parameters
        ----------
        device :
            device where the net will run
        loss_function_name : str
            loss_function name
        optimizer_name: str
            optimizer name
        """

        if loss_function_name is None:
            loss_function_name = self.l_conf_data['loss_function']
        loss_function_constructor = self.get_loss_function_by_name(loss_function_name)
        self.loss_function = loss_function_constructor()
        if optimizer_name is None:
            optimizer_name = self.l_conf_data['optimizer']
        optimizer_constructor = self.get_optimizer_by_name(optimizer_name)
        self.optimizer = optimizer_constructor(self.parameters(), self.lr)
        self.to(device)
        print(f"Net will run in {self.device}")

    def forward(self, x):
        """
        Method that forwards the data thorugh the net. Will be called everytime the net is called with
        a data parameter. (Example: net(data) )

        Parameters
        ----------
        x :
            data to be forwarded

        Returns
        -------
        x :
            data after being forwarded
        """
        x = self.forward_data_through_layers(self.conv_layers, x)
        x = x.view(-1, self._flatten_dim)
        return self.forward_data_through_layers(self.dense_layers, x)  # Activation Function

    def make_predictions(self):
        """
        Method that makes predictions based on the predictions folder. Will show each image with the
        predicted class as its title
        """
        predict_data_path = self.data_loader.predict_dir_path
        for file in os.listdir(predict_data_path):
            try:
                img = self.data_loader.read_image(predict_data_path / file)
                img = torch.Tensor(img).view(self.data_loader.get_input_size()).to(self.device)
                img = self.data_loader.normalize_img(img)
                output = torch.argmax(self(img))
                output = self.data_loader.classes[output]
                self.data_loader.show_image(predict_data_path / file, output)

            except Exception as e:
                print(e)
                print(f"{file} could not be loaded")


def optimize(device, img_loader, loss_functions_names=None, optimizers_names=None, batch_sizes=None, lrs=None,
             epochs=None,
             log_file=None):
    """
    Function that creates all possible different combinations of nets with the parameters given and
    logs its results in order to view which are the best hyper-parameters

    Parameters
    ----------
    device :
        device where the net will be running
    img_loader :
        data_loader for the net
    loss_functions_names : list
        list with all the loss_functions we want to try
    optimizers_names : list
        list with all the optimizers we want to try
    batch_sizes : list
        list with all the batches we want to try
    lrs : list
        list with all the learning rates we want to try
    epochs : list
        list with all the max_epochs we want to try
    log_file : str
        filename where we will store all the log data of the different created nets
    """

    if log_file is None:
        log_file = f"optimizer_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"

    l_conf_data = read_conf('/config/ImgConvNet_conf.json')
    if lrs is None:
        lrs = l_conf_data['optimizer_defaults']['lrs']
    if batch_sizes is None:
        batch_sizes = l_conf_data['optimizer_defaults']['batch_sizes']
    if epochs is None:
        epochs = l_conf_data['optimizer_defaults']['epochs']
    if loss_functions_names is None:
        loss_functions_names = l_conf_data['optimizer_defaults']['loss_functions']
        # loss_functions = [loss_dict[x]() for x in loss_functions]

    if optimizers_names is None:
        optimizers_names = l_conf_data['optimizer_defaults']['optimizers']
        # optimizers_constructors = [optimizer_dict[x] for x in optimizers_constructors]

    i = 0
    for epoch in epochs:
        for batch_size in batch_sizes:
            for loss_function_name in loss_functions_names:
                for lr in lrs:
                    for optimizer_name in optimizers_names:
                        net = ConvNet(device=device, img_loader=img_loader, loss_function_name=loss_function_name,
                                      optimizer_name=optimizer_name, lr=lr)
                        net.train_p(batch_size=batch_size, max_epochs=epoch,
                                    model_name=f"{optimizer_name}_{lr}_{loss_function_name}_{epoch}_{batch_size}",
                                    log_file=log_file, verbose=True)
                        i += 1

    print(f"Trained: {i} Different models")
