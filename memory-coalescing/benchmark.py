import os
import sys

import matplotlib.pyplot as plt
import seaborn

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT)
from utils import benchmark


keys = ["Start offset", "Move offset"]
values = [
    (1, 32),
    (32, 1),
]

frame = benchmark(keys, values)

g = seaborn.barplot(data=frame, x="Start offset", y="Time", hue="Move offset")
plt.show()
