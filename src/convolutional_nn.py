import os  # OS library
import cv2  # library of computer-vision functions
from skimage.io import imread  # Scikit image library: Reads an image
from skimage.transform import resize  # Scikit image library: Resizes an image
import numpy as np  # Vector - Matrix Library
from tqdm import tqdm  # Progress bar library
from pathlib import Path  # Path manipulation
import matplotlib.pyplot as plt  # Graph making Library

# Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False


class DogsVSCats():
    DATA_BASE_DIR = Path("../data")
    IMG_SIZE = 50
    CATS = DATA_BASE_DIR / "cats_dogs/PetImages/Cat"
    DOGS = DATA_BASE_DIR / "cats_dogs/PetImages/Dog"
    TRAINING_DATA_PATH = DATA_BASE_DIR / "training_data.npy"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    cat_count = 0
    dog_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            error = 0
            for file in tqdm(os.listdir(label)):
                try:
                    path = label / file
                    # img = imread(path, as_gray=True)
                    # img = resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # NOTE: cv2 does not work properly with Path library paths
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # np.eye(N) returns the identity matrix NxN and can be used to get One-hot vectors
                    one_hot = np.eye(len(self.LABELS))[self.LABELS[label]]
                    self.training_data.append([np.array(img), one_hot])

                    if label == self.CATS:
                        self.cat_count += 1
                    elif label == self.DOGS:
                        self.dog_count += 1
                except Exception as e:
                    error += 1
                    print(e)
            print("ERRORS: ", error)
        np.random.shuffle(self.training_data)
        np.save(self.TRAINING_DATA_PATH, self.training_data)
        print("Cats: ", self.cat_count)
        print("Dogs: ", self.dog_count)


class Net(nn.Module):
    def __init__(self, img_h=50, img_w=None):
        super().__init__()
        self.img_h = img_h

        if img_w is None:
            self.img_w = img_h
        else:
            self.img_w = img_w

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
        x = torch.rand(self.img_h, self.img_w).view(-1, 1, self.img_h, self.img_w)
        x = self.convs(x, False)
        # print(x[0].shape)
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


# print(training_data[1][1])
# plt.imshow(training_data[1][0], cmap="gray")
# plt.show()

def prepare_Xy(path, val_pct, img_h, img_w):
    training_data = np.load(path, allow_pickle=True)
    X = [i[0] for i in training_data]
    X = torch.Tensor(X).view(-1, img_h, img_w)
    X = X / 255.0
    y = [i[1] for i in training_data]
    y = torch.Tensor(y)

    val_size = int(len(X) * val_pct)

    train_X = X[:-val_size]
    train_y = y[:-val_size]
    test_X = X[-val_size:]
    test_y = y[-val_size:]

    return train_X, test_X, train_y, test_y


def train(train_X, train_y, net, batch_size, epochs, optimizer, loss_function, img_h, img_w):
    for epoch in range(epochs):
        for i in tqdm(range(0, len(train_X), batch_size)):
            # print(i, i+BATCH_SIZE)
            batch_X = train_X[i:i + batch_size].view(-1, 1, img_h, img_w)
            batch_y = train_y[i:i + batch_size]

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}. Loss: {loss}")


def test(test_X, test_y, net, img_h, img_w):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, img_h, img_w))[0]
            predicted_class = torch.argmax(net_out)
            if (predicted_class == real_class):
                correct += 1
            total += 1
    print("Accuracy:", round(correct / total, 3))


LR = 0.001
VAL_PCT = 0.1
BATCH_SIZE = 100
EPOCHS = 1

# Execution
dogvscats = DogsVSCats()
if REBUILD_DATA:
    dogvscats.make_training_data()

net = Net(dogvscats.IMG_SIZE)
optimizer = optim.Adam(net.parameters(), LR)
loss_function = nn.MSELoss()

train_X, test_X, train_y, test_y = prepare_Xy(dogvscats.TRAINING_DATA_PATH, VAL_PCT,
                                              dogvscats.IMG_SIZE, dogvscats.IMG_SIZE)
train(train_X, train_y, net, BATCH_SIZE, EPOCHS,
      optimizer, loss_function, dogvscats.IMG_SIZE, dogvscats.IMG_SIZE)

test(test_X, test_y, net, dogvscats.IMG_SIZE, dogvscats.IMG_SIZE)
