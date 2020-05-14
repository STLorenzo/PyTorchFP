import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose({
                           transforms.ToTensor()
                       }))

test = datasets.MNIST('', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # F.relu() activation function
        # fc1() is a nn.Linear() that feedforwards data in the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()
# print(net)

X = torch.rand((28, 28))
# The -1 means that expected any size
# We have to pass how many samples are going to be and their dim
X = X.view(-1, 28 * 28)
output = net(X)
# print(output)

# lr=0.001 <=> learning rate = 1e-3  
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featuresets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28 * 28))
        loss = F.nll_loss(output, y)
        # Back propagation
        loss.backward()
        # update weigths
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))
plt.imshow(X[0].view(28, 28))
plt.show()

print(torch.argmax(net(X[0].view(-1, 28 * 28))[0]))

