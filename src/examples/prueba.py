import signal
import sys
import numpy as np
from collections import Counter
from src.general_functions import read_conf
import torch

# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
#
#
# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# signal.pause()

# def metodo(perro='a', gato='a', pez='a'):
#     print(f"El perro {perro}")
#     print(f"El gato {gato}")
#     print(f"El pez {pez}")
#
#
# dic = {'pez': 'Chispas'}
#
# metodo("manolo", **dic)

x = torch.rand(4,4)
print(x)
size = (-1,2,2)
y = x.view(-1,2,2)
print(y)
z = x.view(size)
print(z)