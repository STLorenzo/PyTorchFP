import cv2  # library of computer-vision functions
import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
# Torch Libraries
import torch
# Personal libraries
from src.general_functions import *
from src.DataLoader import DataLoader


class ImageLoader(DataLoader):
    """
    Class which reads a folder that contains N_class subfolders containing images of that class. Each subfolder has
    to be named like the class to be predicted. This class preprocesses all the images to be in a suitable input form
    for a neural net in torch.Tensors datatypes. It also can preprocess a folder that contains images from all the
    classes for the net to make predictions.

    Attributes
    ----------
    conf_filename : str
        path to the class configuration file

    project_conf_filename : str
        path to the project configuration file

    self.img_dir_path : Path
        path to the folder that contains all the subfolders with the images
    self.predict_dir_path : Path
        path to the folder that contains the images for prediction
    self.img_size : (int, int)
        tuple with the (img_height, img_width) values
    self.img_norm_value : float
        images normalization value. (usually 255.0)

    self.base_path : Path
        path to the root of the project
    self.data_base_path : Path
        path to the data folder
    self.created_data_path : Path
        path to the folder that will contain the data created by this class
    self.training_data_dir_path : Path
        path to the folder that stores the training_test data preprocessed
    self.training_data_filename : str
        filename of the file that stores the training_test data

    self.classes : list
        list with the names of all the classes
    self.labels : dict
        dictionary with the name of classes as key and their respective label as value
    self.training_data : list
        list to initially store all the training data
    self.counts : dict
        dictionary with the name of classes as key and their total amount of samples as value

    """

    def __init__(self, img_dir_path=None, predict_dir_path=None, img_size=None, img_norm_value=None):
        """
        Constructor for the class

        Parameters
        ----------
        img_dir_path : Path
            path to the folder containing the subfolders of each class with images
        predict_dir_path : Path
            path to the folder with images from all the classes mixed to be predicted
        img_size : (int, int)
            size for the image to be resized
        img_norm_value : float
            value used for the normalization of the images. Usually 255.0
        """

        self.conf_filename = "/config/ImageLoader_conf.json"
        self.project_conf_filename = "/config/Project_conf.json"

        p_conf_data = read_conf(self.project_conf_filename)
        l_conf_data = read_conf(self.conf_filename)

        self.img_dir_path = img_dir_path
        self.predict_dir_path = predict_dir_path
        self.img_size = img_size
        self.img_norm_value = img_norm_value

        self.base_path = Path(p_conf_data['base_path'])
        self.data_base_path = self.base_path / p_conf_data['dirs']['data_dir']
        self.created_data_path = self.data_base_path / l_conf_data['dirs']['created_data_dir']
        self.training_data_dir_path = self.created_data_path / l_conf_data['dirs']['training_test_data_dir']
        self.training_data_filename = l_conf_data['filenames']['training_data_numpy']

        if self.img_dir_path is None:
            self.img_dir_path = self.data_base_path / l_conf_data['dirs']['img_dir']
        if self.predict_dir_path is None:
            self.predict_dir_path = self.data_base_path / l_conf_data['dirs']['predict_dir']

        if self.img_size is None:
            self.img_size = (l_conf_data["img_sizes"]['img_h'], l_conf_data["img_sizes"]['img_w'])
        if self.img_norm_value is None:
            self.img_norm_value = l_conf_data['img_norm_value']

        create_dir(self.created_data_path)
        create_dir(self.training_data_dir_path)

        self.classes = os.listdir(self.img_dir_path)
        self.labels = {label: i for label, i in zip(self.classes, range(len(self.classes)))}
        create_dir(self.created_data_path)
        self.training_data = []
        self.counts = {label: 0 for label in self.classes}

    def read_image(self, path):
        """
        Reads an image from the path given and returns a resized version of it

        Parameters
        ----------
        path
            The path to the image to be read

        Returns
        -------
        The image resized
        """
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (self.img_size[0], self.img_size[1]))

    def show_image(self, path, title="None"):
        """
        Shows the image given in a window using the opencv library

        Parameters
        ----------
        path : Path
            path to the image
        title : str
            title for the window

        """
        img = cv2.imread(str(path))
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)  # cv2 has problem Ubuntu

    def normalize_img(self, img):
        """
        Normalized an image establishing all of its values to floats between 0 and 1

        Parameters
        ----------
        img
            image to be normalized

        Returns
        -------
            the image normalized

        """
        return img / self.img_norm_value

    def one_hot_to_list(self, matrix):
        """
        Given a list of one-hot encodings returns a list of int with each corresponding label value
        Parameters
        ----------
        matrix
            list of one-hot encodings

        Returns
        -------
            list of int
        """
        values = []
        for row in matrix:
            identity = np.eye(len(row))
            for i, row_i in enumerate(identity):
                if (row_i == np.array(row)).all():
                    values.append(i)
        return values

    def make_data(self, val_pct=None):
        """
        Reads all the input images of each class and assigns their corresponding class label. Transforms each
        image into a torch Tensor and resizes it to be suitable for input for a neural net. Finally saves the data
        on disk.

        Parameters
        ----------
        val_pct
            percentage expressed between 0 and 1 of the part of the data to be used as validation data

        """
        if val_pct is None:
            l_conf_data = read_conf(self.conf_filename)
            val_pct = l_conf_data['val_pct']
        self.training_data = []
        self.counts = {label: 0 for label in self.classes}
        for label in self.labels:
            print(f"Reading {label} images...")
            errors = 0
            for file in tqdm(os.listdir(self.img_dir_path / label)):
                try:
                    path = self.img_dir_path / label / file
                    # img = imread(path, as_gray=True)
                    # img = resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # NOTE: cv2 does not work properly with Path library paths
                    img = self.read_image(path)
                    # np.eye(N) returns the identity matrix NxN and can be used to get One-hot vectors
                    one_hot = np.eye(len(self.labels))[self.labels[label]]
                    self.training_data.append([np.array(img), one_hot])
                    self.counts[label] += 1
                except Exception as e:
                    errors += 1
            print("Files not loaded due to errors: ", errors)
        np.random.shuffle(self.training_data)
        np.save(self.training_data_dir_path / self.training_data_filename, self.training_data)
        print("Amount of data for each class:")
        for k, v in self.counts.items():
            print(f"{k}: {v}")
        self.prepare_Xy(val_pct)

    def prepare_Xy(self, val_pct=0.1):
        """
        Internal function that transforms the numpy arrays created by the make-data method into torch Tensors
        and saves them in disk.

        Parameters
        ----------
        val_pct
            percentage expressed between 0 and 1 of the part of the data to be used as validation data

        """
        print("Creating train_X, train_y, test_X, test_y...")
        training_data = np.load(self.training_data_dir_path / self.training_data_filename, allow_pickle=True)
        X = [i[0] for i in training_data]
        X = torch.Tensor(X).view(-1, self.img_size[0], self.img_size[1])
        X = self.normalize_img(X)
        y = [i[1] for i in training_data]
        y = torch.Tensor(y)

        val_size = int(len(X) * val_pct)

        train_X = X[:-val_size]
        train_y = y[:-val_size]
        test_X = X[-val_size:]
        test_y = y[-val_size:]

        self.save_Xy(train_X, train_y, test_X, test_y)
        print("Data saved")

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
        torch.save(train_X, self.training_data_dir_path / "train_X.pt")
        torch.save(train_y, self.training_data_dir_path / "train_y.pt")
        torch.save(test_X, self.training_data_dir_path / "test_X.pt")
        torch.save(test_y, self.training_data_dir_path / "test_y.pt")

    def read_train_X(self):
        return torch.load(self.training_data_dir_path / "train_X.pt")

    def read_train_y(self):
        return torch.load(self.training_data_dir_path / "train_y.pt")

    def read_test_X(self):
        return torch.load(self.training_data_dir_path / "test_X.pt")

    def read_test_y(self):
        return torch.load(self.training_data_dir_path / "test_y.pt")

    def get_input_size(self):
        """
        Returns the size (shape) that the tensor has to have for being input to the net

        Returns
        -------
        tuple
            tuple with the tensor shape for the input to the net
        """
        return -1, 1, self.img_size[0], self.img_size[1]

    def get_image_size(self):
        """
        Returns the image resize shape

        Returns
        -------
        tuple
            tuple with the image resize shape
        """
        return self.img_size
