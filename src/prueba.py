import signal
import sys
import numpy as np
from collections import Counter

# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
#
#
# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# signal.pause()
x = []
for i in range(1000):
 x.append(np.random.randint(0,2))

print(Counter(x))