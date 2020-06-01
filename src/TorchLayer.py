# Torch Libraries
import torch.nn as nn


class TorchLayer(nn.Module):
    """
    Class that represents a layer of the TorchNet module. Instantiates a given torch.layer constructor
    and stores the activation functions with their parameters for being applied when the data is being
    forwarded.

    Attributes
    ----------
    constructor : torch Module
        constructor of the torch layer to be instantiated
    params_dict : dict
        dictionary with the parameters for the constructor
    functions_wp : list
        list containing tuples in which the first element is the activation function constructor and the
        second is a dict with the parameters to be passed to the function
    layer
        the layer module instantiated
    """
    def __init__(self, constructor, params_dict):
        """
        Constructor for the class

        Parameters
        ----------
        constructor : torch Module
            constructor of the torch layer to be instantiated
        params_dict : dict
            dictionary with the parameters for the constructor
        """
        super().__init__()
        self.constructor = constructor
        self.params_dict = params_dict
        self.functions_wp = []
        self.layer = self.construct()

    def construct(self):
        """
        constructs the layer using the dictionary with its parameters

        Returns
        -------
        constructor:
            the torch layer instantiated
        """
        return self.constructor(**self.params_dict)

    def add_function_with_parameters(self, constructor, params=None):
        """
        Appends an activation function and its dictionary of parameters
        to the list of them stored by the class

        Parameters
        ----------
        constructor
            activation function constructor
        params
            dictionary of parameters for the activation function
        """
        if params is None:
            params = {}
        self.functions_wp.append((constructor, params))

    def forward_data(self, x):
        """
        Given a data input x forwards its data through the player
        Parameters
        ----------
        x : torch Tensor
            data to be forwarded by the layer

        Returns
        -------
        x :
            output of the data after being forwarded
        """
        x = self.layer(x)
        for f_wp in self.functions_wp:
            function = f_wp[0]
            params = f_wp[1]
            x = function(x, **params)
        return x
