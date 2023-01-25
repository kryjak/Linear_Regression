"""Polynomial regression"""

"""
Polynomial regression, i.e. fitting y_n = b0 + b1*x + b2*x^2 + ... + bn*x^n can be seen as a special case of multiple 
linear regression where x, x^2, ..., x^n are treated as the independent variables.
In a way, this is still a linear model, because it's linear in the coefficients bi that we need to determine.
So we can use the same techniques as in the genuine multi-dimensional regression.
https://en.wikipedia.org/wiki/Polynomial_regression
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# load the data and turn into a numpy array
data_file = '/home/jkrys/Desktop/Programming/machine_learning_examples/linear_regression_class/data_poly.csv'

fo = open(data_file)
data = np.array(list(csv.reader(fo)), dtype='float')
fo.close()

print(f'N: {np.shape(data)[0]}, D: {np.shape(data)[1] - 1}')
Xmat, Y = data[:, 0], data[:, 1]

# we need to prepend a column of 1s to Xmat to account for the b0 term that has been absorbed into w
# nice trick: https://stackoverflow.com/questions/8486294/how-do-i-add-an-extra-column-to-a-numpy-array
Xmat = np.c_[np.ones(Xmat.shape[0]), Xmat, Xmat**2]
# alternatively, we can concatenate an (100, 1) array of ones along axis 1 of Xmat
# Xmat = np.concatenate((np.ones((Xmat.shape[0], 1)), Xmat), axis=1)
print(Xmat.shape)

w = np.linalg.solve(np.matmul(np.transpose(Xmat), Xmat), np.matmul(np.transpose(Xmat), Y))

def Yhat(x):
    return np.matmul(np.transpose(w), x)

SSres = np.sum((Y - Yhat(np.transpose(Xmat))) ** 2)
SStot = np.sum((Y - np.mean(Y)) ** 2)
Rsq = 1 - SSres/SStot

def test(x):
    return np.matmul(np.array([1, 2]), x)

print(f'R^2 is {Rsq}')

xs = np.linspace(Xmat[:, 1].min(), Xmat[:, 1].max())
xs = np.c_[np.ones(xs.shape[0]), xs, xs**2]

plt.scatter(Xmat[:, 1], Y)
plt.plot(xs[:, 1], Yhat(np.transpose(xs)))

plt.title('Polynomial regression')
plt.xlabel('x')
plt.ylabel('y')

# multiline text that is compatible with LaTeX:
# https://stackoverflow.com/a/37930579/7799311
params = f'$b_0 = {round(w[0], 3)}$\n' + f'$b_1 = {round(w[1], 3)}$\n' + f'$b_2 = {round(w[2], 3)}$\n' + \
         f'$R^2 = {round(Rsq, 3)}$'
plt.text(0, 800, params, bbox={'facecolor': 'blue', 'alpha': 0.2})

print(w)
plt.show()
