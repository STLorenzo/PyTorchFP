from abc import ABC, abstractmethod


class DataLoader(ABC):
    """
    Abstract class for implementing a class that makes, saves and loads data for TorchNet

    Attributes
    ----------
    conf_filename : str
        path to the class configuration file
    project_conf_filename : str
        path to the project configuration file

    """
    conf_filename = None
    project_conf_filename = None

    @abstractmethod
    def make_data(self, val_pct=None):
        """
        Creates and stores the data prepared in training and test subsets

        Parameters
        ----------
        val_pct : float
            percentage of the data separated for validation. Value has to be between 0 and 1
        """
        pass

    @abstractmethod
    def save_Xy(self, train_X, train_y, test_X, test_y):
        """
        Saves each tensor in a file

        Parameters
        ----------
        train_X : torch.Tensor
            torch tensor with the input data for training
        train_y : torch.Tensor
            torch tensor with the output data for training
        test_X : torch.Tensor
            torch tensor with the input data for testing
        test_y : torch.Tensor
            torch tensor with the output data for testing
        """
        pass

    def read_XY(self):
        """
        Reads the training and test data separated in input and output and returns a tensor for each one of them

        Returns
        -------
        train_X : torch.Tensor
            torch tensor with the input data for training
        train_y : torch.Tensor
            torch tensor with the output data for training
        test_X : torch.Tensor
            torch tensor with the input data for testing
        test_y : torch.Tensor
            torch tensor with the output data for testing

        """
        print("Loading train_X, train_y, test_X, test_y...")
        train_X = self.read_train_X()
        train_y = self.read_train_y()
        test_X = self.read_train_X()
        test_y = self.read_train_y()
        print("Load successful")
        return train_X, test_X, train_y, test_y

    @abstractmethod
    def read_train_X(self):
        pass

    @abstractmethod
    def read_train_y(self):
        pass

    @abstractmethod
    def read_test_X(self):
        pass

    @abstractmethod
    def read_test_y(self):
        pass

    @abstractmethod
    def get_input_size(self):
        """
        Returns the size (shape) that the tensor has to have for being input to the net

        Returns
        -------
        tuple
            tuple with the tensor shape for the input to the net
        """
        pass
