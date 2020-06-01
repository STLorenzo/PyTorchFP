import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
import time  # Time measuring library
import datetime
import signal
# Torch Libraries
import torch
import torch.nn as nn
# Personal Libraries
from src.general_functions import *
from abc import ABC, abstractmethod


class TorchNet(nn.Module, ABC):
    """
    Abstract class for implementing a class that represents a torch neural network.

    Attributes
    ----------
    STOP_TRAIN : bool
        Flag for early stopping training and save it's instance so it can be resumed later.
    loss_dict : dict
        dictionary with all the loss functions supported by the net
    optimizer_dict : dict
        dictionary with all the optimizers supported by the net
    conf_path : Path
        path to the class configuration file
    self.project_conf_path : Path
        path to the project congifuration file
    self.p_conf_data : dict
        dictionary obtained from reading the project configuration file
    self.l_conf_data : dict
        dictionary obtained from reading the class configuration file
    self.data_loader : DataLoader
        DataLoader class for importing the data to the net
    self.device :
        device in which the net is going to run
    self.lr : float
        learning rate of the net

    self.optimizer :
        net optimizer
    self.loss_function :
        net loss function

    self.MAX_VAL_TRAIN_PCT :
        maximum percentage of training data the net can use for validation during training step
    self.val_train_pct :
        percentage of training data the net will use for validation during training step

    self.base_path : Path
        path to the root of the project
    self.data_base_path : Path
        path to the data folder
    self.created_data_path : Path
        path to the folder that will contain the data created by this class
    self.models_path : Path
        path to the folder where the models will be stored
    self.half_trained_model_path : Path
        path where the half trained models will be stored
    self.logs_path : Path
        path where the log files from the training will be stored
    """

    def __init__(self, data_loader, device=None, lr=None):
        super().__init__()
        self.STOP_TRAIN = False
        self.loss_dict = None
        self.optimizer_dict = None
        self.conf_path = None
        self.project_conf_path = None
        self.p_conf_data = None
        self.l_conf_data = None

        self.data_loader = data_loader
        self.device = device
        self.lr = lr

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

    def assign_conf_data(self):
        """
        Method that reads the configuration data and asigns its values to the internal variables.
        """
        if self.project_conf_path is None or self.conf_path is None:
            raise Exception("Filenames not defined")

        self.p_conf_data = read_conf(self.project_conf_path)
        self.l_conf_data = read_conf(self.conf_path)

    def assign_paths(self):
        """
        Method that assigns the paths read from the configuration file to the internal variables
        """
        if self.p_conf_data is None or self.l_conf_data is None:
            raise Exception("Conf data not defined")

        # TODO: rename p_conf_data and l_conf_data
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
        """
        Method that prepares signal handlers for half stopping the training
        """
        signal.signal(signal.SIGINT, self.catch_abrupt_end)
        signal.signal(signal.SIGTERM, self.catch_abrupt_end)

    def catch_abrupt_end(self, signum, frame):
        """
        function to be executed when the stopping program signals are catch

        Parameters
        ----------
        signum
        frame

        """
        self.STOP_TRAIN = True

    @abstractmethod
    def establish_architecture(self):
        """
        Method that prepares the net architecture establishing all its layers with their corresponding
        activation functions
        """
        pass

    @abstractmethod
    def compile_net(self, device, loss_function_name, optimizer_name):
        """
        Method tha compiles the net, effectively assigning a loss function, optimizer and device for the net.

        Parameters
        ----------
        device :
            device where the net is going to be executed
        loss_function_name : str
            loss function name that has to be in the class loss_function dictionary so its constructor can be used.
        optimizer_name : str
            optimizer name that has to be in the class optimizer dictionary so its constructor can be used.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Method that forwards x data through the net
        Parameters
        ----------
        x :
            data for input to the net
        """
        pass

    def fwd_pass(self, X, y, loss_function=None, optimizer=None, train=False):
        """
        Method that forwards the data, calculates its error with the loss function and backpropagates it
        so the net can be optimized with the optimizer
        Parameters
        ----------
        X :
            features of the dataset
        y :
            target of the dataset
        loss_function :
            loss_function to be used in this step
        optimizer :
            optimizer to be used in this step
        train : bool
            boolean value that indicates if we are in a training process so the optimizer step is produced

        Returns
        -------
        acc : float
            accuracy of this step
        loss : float
            error of this step
        """
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

    def train_p(self, train_X=None, train_y=None, batch_size=None, epoch=0, max_epochs=None, log_file=None,
                loss_function=None, val_train_pct=None,
                optimizer=None, model_name=f"model-{time.time()}", n_steps_log=None, verbose=False):
        """
        Method that trains the net.
        Parameters
        ----------
        train_X :
            features of the training data
        train_y :
            target of the training data
        batch_size :
            size of the batch for each forward step
        epoch : int
            starting epoch
        max_epochs : int
            max amount of epochs
        log_file : str
            filename of the file where the logs are going to be written
        loss_function :
            loss function to be used
        val_train_pct : float
            amount of training data to be used for validation and not be trained
        optimizer :
            optimizer to be used
        model_name : str
            The name of the model that is being trained
        n_steps_log : int
            number of iterations that have happen for the training status to be logged
        verbose : bool
            flag to indicate if status messages are printed in strdout

        """
        # Input check
        if val_train_pct is None:
            val_train_pct = self.val_train_pct

        if val_train_pct > self.MAX_VAL_TRAIN_PCT or val_train_pct < 0:
            raise Exception(f"train_p error: val_train_pct higher than max({self.MAX_VAL_TRAIN_PCT}) or lower than 0")
        if epoch >= max_epochs:
            raise Exception(f"max_epochs lower than the starting epoch max_epoch: {max_epochs} - epoch: {epoch}")

        # Load train data if not given
        if train_X is None:
            train_X = self.data_loader.read_train_X()

        if train_y is None:
            train_y = self.data_loader.read_train_y()

        test_X = None
        test_y = None

        # if valid validation percentage given separate the corresponding training data for validation
        if val_train_pct > 0:
            pct_index = int(val_train_pct * len(train_X))
            point = np.random.randint(len(train_X) - pct_index)
            test_X = train_X[point:point + pct_index]
            test_y = train_y[point:point + pct_index]
            train_X = torch.cat((train_X[:point], train_X[point + pct_index:]), 0)
            train_y = torch.cat((train_y[:point], train_y[point + pct_index:]), 0)

        # Default log file name with the timestamp of the moment so it doesn't override.
        # Timestamp given in the form YY-MM-DD_hh_mm to allow filename compatibility with windows
        if log_file is None:
            log_file = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')}.log"

        if batch_size is None:
            batch_size = self.l_conf_data['batch_size']
        if max_epochs is None:
            max_epochs = self.l_conf_data['max_epochs']
        if n_steps_log is None:
            n_steps_log = self.l_conf_data['n_steps_log']

        log_file_path = self.logs_path / log_file

        # If loss_function or optimizer not given use the net default ones
        loss_function, optimizer = self.check_optim_loss(loss_function, optimizer)

        optimizer_name, lr = self.get_optimizer_data(optimizer)
        loss_function_name = self.get_loss_function_name(loss_function)

        # Flag for being able to stop trainning half way and save the instance for future resuming
        self.STOP_TRAIN = False

        if verbose:
            print(f"Starting Training of {model_name},{optimizer_name},{lr},"
                  f"{loss_function_name},{max_epochs},{batch_size}")

        t0 = time.time()
        with open(log_file_path, "a") as f:
            for epoch in range(epoch, max_epochs):
                # If flag is active because it received a SIGINT OR SIGTERM save the net instance and exit the program
                if self.STOP_TRAIN:
                    if verbose:
                        print("Stopping Training")
                    # self.print_instance(epoch, max_epochs, batch_size, optimizer, loss_function)
                    self.save_instance_net(self.half_trained_model_path / f"__half__{model_name}.pt",
                                           epoch, max_epochs, batch_size, optimizer, loss_function,
                                           log_file, model_name)
                    sys.exit(0)

                # Training procedure
                print(f"Epoch: {epoch + 1}/{max_epochs}")
                for i in tqdm(range(0, len(train_X), batch_size)):
                    # Separate data in batches
                    batch_X = train_X[i:i + batch_size].view(self.data_loader.get_input_size()).to(self.device)
                    batch_y = train_y[i:i + batch_size].to(self.device)
                    # Forward the data
                    acc, loss = self.fwd_pass(batch_X, batch_y, loss_function, optimizer, train=True)
                    # After n amount of steps perform a test to log the training evolution
                    if i % n_steps_log == 0:
                        self.train_test(batch_X, batch_y, test_X, test_y, loss_function, optimizer, val_train_pct,
                                        f, model_name, epoch, loss_function_name, optimizer_name, lr, batch_size, acc,
                                        loss)

            # Final test once training has finished
            self.train_test(batch_X, batch_y, test_X, test_y, loss_function, optimizer, val_train_pct,
                            f, model_name, epoch, loss_function_name, optimizer_name, lr, batch_size, acc,
                            loss)

        t1 = time.time() - t0
        if verbose:
            print(f"Training Finished in {t1} seconds")

    def resume_training(self, path):
        """
        Resumes a training that was stopped halfway

        Parameters
        ----------
        path : Path
            path to the model instance saved

        """
        epoch, max_epoch, loss_function, batch_size, log_file, model_name = self.load_instance_net(path)
        self.train_p(epoch=epoch, max_epochs=max_epoch, batch_size=batch_size, loss_function=loss_function,
                     optimizer=self.optimizer, log_file=log_file, model_name=model_name, verbose=True)

    def train_test(self, batch_X, batch_y, test_X, test_y, loss_function, optimizer, val_train_pct,
                   f, model_name, epoch, loss_function_name, optimizer_name, lr, batch_size, acc, loss):
        """
        Methods that performs a test during the training phase.

        Parameters
        ----------
        batch_X :
            batch of features to be tested
        batch_y :
            targets of the batch
        test_X :
            test feature data
        test_y :
            test target data
        loss_function :
            loss function to be used
        val_train_pct : float
            amount of training data to be used for validation and not be trained
        optimizer :
            optimizer to be used
        f : File
            opened file where data will be written
        model_name : str
            The name of the model that is being trained
        epoch : int
            current epoch
        loss_function_name : str
            name of the loss_function
        optimizer_name : str
            name of the optimizer
        lr : float
            learning rate of the model
        batch_size :
            size of the batch for each forward step
        acc : float
            accuracy of the training step
        loss : float
            loss value of the training step

        Returns
        -------
        val_acc : float
            accuracy of the test
        val_loss : float
            loss value os the test
        """
        if val_train_pct > 0:
            # If the validation_pct is more than 0 means that we have test data and we have
            # to create batches for it instead of using the default train batches
            i = np.random.randint(len(test_X) - batch_size)
            batch_X = test_X[i:i + batch_size].view(self.data_loader.get_input_size()).to(self.device)
            batch_y = test_y[i:i + batch_size].to(self.device)
        val_acc, val_loss = self.test_p(batch_X, batch_y, loss_function, optimizer)

        f.write(f"{model_name},{epoch},{round(time.time(), 3)},"
                f"{loss_function_name},{optimizer_name},{lr},{batch_size},"
                f"{round(float(acc), 2)},{round(float(loss), 4)},"
                f"{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")

    def test_p(self, test_X=None, test_y=None, loss_function=None, optimizer=None, size=None, verbose=False):
        """
        Method that performs a test in the data

        Parameters
        ----------
        test_X :
            test feature data
        test_y :
            test target data
        loss_function :
            loss function to be used
        optimizer :
            optimizer to be used
        size : int
            size of the test data bath to be tested
        verbose : bool
            flag to indicate if status messages are printed in strdout

        Returns
        -------
        val_acc : float
            accuracy of the test
        val_loss : float
            loss value os the test
        """
        if test_X is None:
            test_X = self.data_loader.read_test_X()
        if test_y is None:
            test_y = self.data_loader.read_test_y()
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
                X.view(self.data_loader.get_input_size()).to(self.device),
                y.to(self.device), loss_function, optimizer)
        t1 = time.time() - t0
        if verbose:
            print(f"Testing Finished in {t1} seconds")
        return val_acc, val_loss

    def check_optim_loss(self, loss_function, optimizer):
        """
        Checks if the net has been compiled, which means the optimizer and the loss_function
        has been established

        Parameters
        ----------
        loss_function :
            loss function to be used
        optimizer :
            optimizer to be used

        Returns
        -------
        loss_function :
            loss function to be used
        optimizer :
            optimizer to be used

        """
        if (optimizer is None and self.optimizer is None) or (loss_function is None and self.loss_function is None):
            raise Exception("Net not compiled")

        if optimizer is None:
            optimizer = self.optimizer
        if loss_function is None:
            loss_function = self.loss_function

        return loss_function, optimizer

    def print_instance(self, epoch, max_epoch, batch_size, optimizer, loss_function):
        """
        prints an instance of the model

        Parameters
        ----------
        epoch : int
            starting epoch
        max_epoch : int
            max amount of epochs
        batch_size :
            size of the batch for each forward step
        loss_function :
            loss function to be used
        optimizer :
            optimizer to be used
        """
        print(f"MODEL: {self.state_dict()}\n"
              f"OPTIMIZER: {optimizer.state_dict()}\n"
              f"Epoch: {epoch}\n"
              f"Max Epoch: {max_epoch}\n"
              f"Batch_size: {batch_size}\n"
              f"Loss: {self.get_loss_function_name(loss_function)}\n")

    def save_instance_net(self, path, epoch, max_epoch, batch_size, optimizer, loss_function,
                          log_file, model_name):
        """
        Saves an instance of the net. Usually during training

        Parameters
        ----------
        path : Path
            path to the file where the instance will be stored
        epoch : int
            starting epoch
        max_epoch : int
            max amount of epochs
        batch_size :
            size of the batch for each forward step
        loss_function :
            loss function to be used
        optimizer :
            optimizer to be used
        log_file : str
            name of the log_file to be used
        model_name : str
            name of the model
        """
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
        """
        Loads an instance of the net

        Parameters
        ----------
        path : Path
            path to the instance file
        """
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
        """
        Saves the net

        Parameters
        ----------
        filename : str
            name of the file where the net will be saved
        """
        if filename is None:
            filename = self.l_conf_data['net_save_default_name']
        path = self.models_path / filename
        torch.save(self.state_dict(), path)

    def load_net(self, filename=None):
        """
        Loads the net

        Parameters
        ----------
        filename : str
            filename for the net to be loaded
        """
        if filename is None:
            filename = self.l_conf_data['net_save_default_name']
        path = self.models_path / filename
        self.load_state_dict(torch.load(path))

    @staticmethod
    def get_optimizer_data(optimizer):
        """
        Returns the optimizer name and learning rate

        Parameters
        ----------
        optimizer :
            optimizer from which to extract the data

        Returns
        -------
        name: str
            optimizer name
        lr : float
            learning rate
        """
        s = str(optimizer)
        name = s.rsplit(' (')[0]
        lr = s.rsplit('lr: ', 1)[1].rsplit('\n')[0]
        return name, lr

    @staticmethod
    def get_loss_function_name(loss_function):
        """
        returns loss function name

        Parameters
        ----------
        loss_function
            loss function from which to extract the data

        Returns
        -------
        name: str
            loss function name
        """
        return str(loss_function)[:-2]

    def get_loss_function_by_name(self, loss):
        """
        Returns a loss_function if its name is in the net loss_function dictionary

        Parameters
        ----------
        loss : str
            loss_function name

        Returns
        -------
        loss_function :
            the loss function

        Raises
        ------
        Exception if the name does not match any key in the dictionary
        """
        if loss in self.loss_dict.keys():
            return self.loss_dict[loss]
        else:
            raise Exception(f"Loss name not doesn't match available functions\n"
                            f"{loss} - {self.loss_dict}")

    def get_optimizer_by_name(self, optimizer_name):
        """
        Returns an optimizer if its name is in the net optimizer dictionary

        Parameters
        ----------
        optimizer_name :str
            the optimizer name

        Returns
        -------
        optimizer:
            the optimizer

        Raises
        ------
            Exception if the name does not match any key in the dictionary
        """
        if optimizer_name in self.optimizer_dict.keys():
            return self.optimizer_dict[optimizer_name]
        else:
            raise Exception(f"Optimizer name not doesn't match available optimizers\n"
                            f"{optimizer_name} - {self.optimizer_dict.keys()}")
