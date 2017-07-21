from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x, y = iris.data[:, :2], iris.target

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=1)

print x.shape, x_train.shape, x_valid.shape

print [x_train[(y_train == i)].shape for i in range(3)]
mu = [x_train[(y_train == i)].mean(axis=0) for i in range(3)]

sigma = [((x_train[(y_train == i)] - mu[i]) ** 2).mean(axis=0) for i in range(3)]
print mu
print sigma


