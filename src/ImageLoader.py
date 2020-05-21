import os  # OS library
import cv2  # library of computer-vision functions
import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
# Torch Libraries
import torch
# Personal libraries
from general_functions import create_dir


class ImageLoader:
    def __init__(self, data_base_dir, img_dir, predict_dir, img_size=(50, 50), img_norm_value=255.0):
        # TODO: Check inputs
        self.data_base_dir = Path(data_base_dir)
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.img_h = img_size[0]
        self.img_w = img_size[1]
        self.classes = os.listdir(self.img_dir)
        self.labels = {label: i for label, i in zip(self.classes, range(len(self.classes)))}
        self.created_data_path = self.data_base_dir / "created_data"
        create_dir(self.created_data_path)
        self.training_data_path = self.created_data_path / "training_data.npy"
        self.training_data = []
        self.counts = {label: 0 for label in self.classes}
        self.img_norm_value = img_norm_value
        self.predict_dir = predict_dir

    def read_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (self.img_size[0], self.img_size[1]))

    def show_image(self, path, output="None"):
        img = cv2.imread(str(path))
        cv2.imshow(output, img)
        cv2.waitKey(0)
        cv2.destroyWindow(output)  # cv2 has problem Ubuntu

    def normalize_img(self, img):
        return img / self.img_norm_value

    def make_training_data(self, val_pct=0.1):
        self.training_data = []
        self.counts = {label: 0 for label in self.classes}
        for label in self.labels:
            print(label)
            errors = 0
            for file in tqdm(os.listdir(self.img_dir / label)):
                try:
                    path = self.img_dir / label / file
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
        np.save(self.training_data_path, self.training_data)
        print("Amount of data for each class:")
        for k, v in self.counts.items():
            print(f"{k}: {v}")
        self.prepare_Xy(val_pct)

    def prepare_Xy(self, val_pct=0.1):
        print("Creating train_X, train_y, test_X, test_y...")
        training_data = np.load(self.training_data_path, allow_pickle=True)
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

        torch.save(train_X, self.created_data_path / "train_X.pt")
        torch.save(train_y, self.created_data_path / "train_y.pt")
        torch.save(test_X, self.created_data_path / "test_X.pt")
        torch.save(test_y, self.created_data_path / "test_y.pt")
        print("Data saved")

        return train_X, test_X, train_y, test_y

    def read_train_X(self):
        return torch.load(self.created_data_path / "train_X.pt")

    def read_train_y(self):
        return torch.load(self.created_data_path / "train_y.pt")

    def read_test_X(self):
        return torch.load(self.created_data_path / "test_X.pt")

    def read_test_y(self):
        return torch.load(self.created_data_path / "test_y.pt")

    def read_Xy(self):
        print("Loading train_X, train_y, test_X, test_y...")
        train_X = self.read_train_X()
        train_y = self.read_train_y()
        test_X = self.read_train_X()
        test_y = self.read_train_y()
        print("Load successful")

        return train_X, test_X, train_y, test_y
