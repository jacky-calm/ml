from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

data, target = load_boston(True)
data = data[:, 0:9]
print data.shape, target.shape
fig = plt.figure('House Price of Boston')

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.scatter(data[:, i], target)

plt.tight_layout()
plt.show()
