# demonstration of L2 regularization
#
# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python

import numpy as np
import matplotlib.pyplot as plt

N = 50

# generate the data
X = np.linspace(0, 10, N)
Y = 2*X + np.random.randn(N)

# make outliers
Y[-1] += 30
Y[-2] += 30

# plot the data
plt.scatter(X, Y)

# add a column of 1s to account for b0 (intercept)
# X = np.array([[1, ii] for ii in X])
X = np.vstack([np.ones(N), X]).T

# ML model
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)

# L2 model
# probably don't need an L2 regularization this high in many problems
# everything in this example is exaggerated for visualization purposes
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml, label='ML estimate')

for l2 in range(10, 1000, 200):
    w_map = np.linalg.solve(l2 * np.identity(2) + X.T.dot(X), X.T.dot(Y))
    Yhat_map = X.dot(w_map)
    plt.plot(X[:, 1], Yhat_map, label=f'MAP, l2: {l2}', alpha=0.5)

plt.legend()
plt.show()
