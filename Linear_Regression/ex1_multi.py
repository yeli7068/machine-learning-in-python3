# %%

#!/anaconda3/bin/python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # for surface plot


# %%
def featureNormaliz(X):
    X_norm = X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)  # ddof=1 除法使用N-1
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    n = np.size(X, axis=1)
    J_history = np.zeros([num_iters, 1])
    for iter in np.arange(1, num_iters + 1):
        delta = 1 / m * (X.T.dot(X).dot(theta) - X.T.dot(y).reshape(n, 1))
        theta = theta - alpha * delta
        J_history[iter - 1] = computeCostMulti(X, y, theta)
    return theta, J_history


def computeCostMulti(X, y, theta):
    m = len(y)
    J = sum((X.dot(theta) - y.reshape(m, 1))**2) / 2 / m
    return J


def normalEqn(X,y):
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y) # pinv(X' * X) * X' * y
    return theta


data = pd.read_csv('ex1data2.txt', header=-1)
data = np.asarray(data)
X = data[:, (0, 1)]
y = data[:, 2]
m = len(y)

#   %% ================ Part 1: Feature Normalization ================

#   对所有特征做归一化

X, mu, sigma = featureNormaliz(X)

#   在第一行添加1，用于thete0的计算
X = np.append(np.ones((m, 1)), X, axis=1)

#   %% ================ Part 2: Gradient Descent ================
alpha = 0.1
num_iters = 50
theta = np.zeros((3, 1))  #  如果要扩展至任意维度也是可以的，这里已经有提示了n

theta, J_history0_01 = gradientDescentMulti(X, y, theta, alpha, num_iters)

alpha = 0.3
num_iters = 50
theta = np.zeros((3, 1))  #  如果要扩展至任意维度也是可以的，这里已经有提示了n

theta, J_history0_03 = gradientDescentMulti(X, y, theta, alpha, num_iters)

alpha =  1
num_iters = 50

theta = np.zeros((3, 1))
theta, J_history0_1 = gradientDescentMulti(X, y, theta, alpha, num_iters)

fig, ax = plt.subplots()

ax.plot(J_history0_01, 'r-')
ax.hold
ax.plot(J_history0_03, 'g-')
ax.hold
ax.plot(J_history0_1, 'b-')

ax.set(xlabel='Number of iterations', ylabel='Cost J')
plt.legend(('alpha = 0.01','alpha = 0.03','alpha = 0.1'))

plt.show()

# %% ================ Part 3: Normal Equations ================

data = pd.read_csv('ex1data2.txt', header=-1)
data = np.asarray(data)
X = data[:, (0, 1)]
y = data[:, 2]
m = len(y)

X = np.append(np.ones((m, 1)), X, axis=1)

theta = normalEqn(X,y)

print(theta)

# %% 