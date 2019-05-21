import os
import sys

import matplotlib.pyplot as plt
import seaborn

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT)
from utils import benchmark


keys = ["Shared memory size"]
values = [0, 16, 512, 2048, 4096, 16384, 24567, 32768]

frame = benchmark(keys, values)

g = seaborn.barplot(data=frame, x="Shared memory size", y="Time")
plt.show()
