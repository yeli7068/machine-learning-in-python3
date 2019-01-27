#!/anaconda3/bin/python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # for surface plot


def computeCost(X, y, theta):
    m = len(y)
    J = 0
    J = sum((X.dot(theta) - y.reshape(m, 1))**2) / 2 / m
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros([num_iters, 1])
    for iter in np.arange(1, num_iters + 1):
        delta = 1 / m * (X.T.dot(X).dot(theta) - X.T.dot(y).reshape(2, 1))
        theta = theta - alpha * delta
        J_history[iter - 1] = computeCost(X, y, theta)
    return theta, J_history


data = pd.read_csv('ex1data1.txt', header=-1)
data = np.asarray(data)
X = data[:, 0]
y = data[:, 1]
m = len(y)

plt.plot(X, y, 'rx')  # Plot Data
X = np.insert(np.ones((m, 1)), 1, X, axis=1)
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta)
print('With theta = [0;0], Expected cost value (approx) 32.07\n')
print(J)
J = computeCost(X, y, np.array([(-1, ), (2, )]))
print('With theta = [-1 ; 2], Expected cost value (approx) 54.24\n')
print(J)

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
plt.plot(X[:, 1], X.dot(theta), "-", label='Training data')
plt.plot(X[:, 1], y, "rx", label='Linear regression')
plt.legend()
plt.figure()
# contour
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

for i in np.arange(0, len(theta0_vals)):
    for j in np.arange(0, len(theta1_vals)):
        t = np.array([(theta0_vals[i], ), (theta1_vals[j], )])
        J_vals[i, j] = computeCost(X, y, t)

plt.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
plt.plot(theta[0], theta[1], 'rx')
plt.show()
# surf

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
plt.show()