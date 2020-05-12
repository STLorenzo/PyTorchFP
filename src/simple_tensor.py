import torch  # machine learning framework using tensors

# Component multiplication
x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])

print(x * y)

x = torch.zeros([2, 5])
print(x)
print(x.shape)

# random value between (0,1) tensor of specified shape
y = torch.rand([2, 5])
print(y)

# .view() <=> .reshape() in numpy
# returns the view but doesn't change the original
print(y.view([1, 10]))

print(y)
