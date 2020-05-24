import os  # OS library
import cv2  # library of computer-vision functions
from skimage.io import imread  # Scikit image library: Reads an image
from skimage.transform import resize  # Scikit image library: Resizes an image
import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
import time  # Time measuring library
import datetime

# Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Creates a directory if it doesn't already exist
def create_dir(dir_name, debug=False):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        if debug:
            print("Directory ", dir_name, " Created ")
    else:
        if debug:
            print("Directory ", dir_name, " already exists")


class ImageLoader():

    def __init__(self, data_base_dir, img_dir, predict_dir, img_size=(50, 50), img_norm_value=255.0):
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


class ImgConvNet(nn.Module):
    def __init__(self, img_loader, device=torch.device('cpu'), optimizer=None, loss_function=None, lr=1e-3):
        super().__init__()
        self.img_loader = img_loader
        self.lr = lr

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

    def train_p(self, train_X=None, train_y=None, batch_size=100, epochs=10, log_file=None, loss_function=None,
                optimizer=None, model_name=f"model-{time.time()}", n_steps_log=50, verbose=False):

        if train_X is None:
            train_X = img_loader.read_train_X()
        if train_y is None:
            train_y = img_loader.read_train_y()

        if log_file is None:
            create_dir("../../doc")
            log_file = Path(f"../doc/{datetime.datetime.now()}.log")

        loss_function, optimizer = self.check_optim_loss(loss_function, optimizer)

        optimizer_name, lr = self.get_optimizer_data(optimizer)
        loss_function_name = self.get_loss_function_name(loss_function)

        if verbose:
            print(f"Starting Training of {model_name},{epochs},"
                  f"{loss_function_name},{optimizer_name},{lr},{batch_size}")
        t0 = time.time()
        with open(log_file, "a") as f:
            for epoch in range(epochs):
                if verbose:
                    print(f"Epoch: {epoch}")
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

    def test_p(self, test_X=None, test_y=None, loss_function=None, optimizer=None, size=None, verbose=False):
        if test_X is None:
            test_X = img_loader.read_test_X()
        if test_y is None:
            test_y = img_loader.read_test_y()
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
        predict_data_path = self.img_loader.predict_dir_path
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

    def save_net(self, path):
        torch.save(self.state_dict(), path)

    def load_net(self, path):
        self.load_state_dict(torch.load(path))

    def get_optimizer_data(self, optimizer):
        s = str(optimizer)
        name = s.rsplit(' (')[0]
        lr = s.rsplit('lr: ', 1)[1].rsplit('\n')[0]
        return name, lr

    def get_loss_function_name(self, loss_function):
        return str(loss_function)[:-2]

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
                    optimizers.append(constructor(net.parameters(), lr))

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
                                     log_file="../../doc/optimizer.log", verbose=True)
                        i += 1

        print(f"Trained: {i} Different models")


# DATA PREPROCESSING
VAL_PCT = 0.2
IMG_SIZE = (50, 50)
# TRAINING PARAMETERS
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 20
# DOC NAMES
MODEL_NAME = f"model-{int(time.time())}"
LOG_FILE = Path(f"../doc/{MODEL_NAME}.log")
# PATHS
DATA_BASE_DIR = Path("../data")
IMG_DIR = DATA_BASE_DIR / "cats_dogs/PetImages"
PREDICT_DIR = DATA_BASE_DIR / "predictions"

# -------------------------------------Execution------------------------------------
REBUILD_DATA = False

img_loader = ImageLoader(DATA_BASE_DIR, IMG_DIR, PREDICT_DIR, IMG_SIZE)
if REBUILD_DATA:
    img_loader.make_training_data(val_pct=VAL_PCT)

MODEL_PATH = img_loader.created_data_path / "net_1.pl"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ImgConvNet(img_loader, DEVICE)
# net.optimize()

# net.train_p(verbose=True, batch_size=BATCH_SIZE, epochs=EPOCHS)
# val_acc, val_loss = net.test_p(verbose=True)
# print("Accuracy: ", val_acc)
# print("Loss: ", val_loss)

# -------- PREDICTIONS ---------------
# net.make_predictions()

# -------- SAVE/LOAD ------------------

# net.save_net(MODEL_PATH)
# net2 = ImgConvNet(img_loader, DEVICE)
# net2.load_net(MODEL_PATH)
# net2.make_predictions()
