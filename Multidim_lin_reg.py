"""
Sample data for multiple linear regression can be found at:
https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# load the data and turn into a numpy array
data_file = '/home/jkrys/Desktop/Programming/machine_learning_examples/linear_regression_class/data_2d.csv'

fo = open(data_file)
data = np.array(list(csv.reader(fo)), dtype='float')
fo.close()

print(f'N: {np.shape(data)[0]}, D: {np.shape(data)[1] - 1}')
Xmat, Y = data[:, :2], data[:, 2]

# we need to prepend a column of 1s to Xmat to account for the b0 term that has been absorbed into w
# nice trick: https://stackoverflow.com/questions/8486294/how-do-i-add-an-extra-column-to-a-numpy-array
Xmat = np.c_[np.ones(Xmat.shape[0]), Xmat]
# alternatively, we can concatenate an (100, 1) array of ones along axis 1 of Xmat
# Xmat = np.concatenate((np.ones((Xmat.shape[0], 1)), Xmat), axis=1)
print(Xmat.shape)

w = np.linalg.solve(np.matmul(np.transpose(Xmat), Xmat), np.matmul(np.transpose(Xmat), Y))

def Yhat(x):
    return np.matmul(np.transpose(w), x)

SSres = np.sum((Y - Yhat(np.transpose(Xmat))) ** 2)
SStot = np.sum((Y - np.mean(Y)) ** 2)
Rsq = 1 - SSres/SStot

print(f'R^2 is {Rsq}')

### plotting
ax = plt.subplot(projection='3d')

# create a grid of x and y values using linspace and meshgrid
# https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
xs = np.linspace(Xmat[:, 1].min(), Xmat[:, 1].max())
ys = np.linspace(Xmat[:, 2].min(), Xmat[:, 2].max())
xs, ys = np.meshgrid(xs, ys)

# reshape so that we can apply Yhat onto each point on the grid
grid = np.reshape(np.transpose([xs, ys]), (50, 50, 2))

# calculate Y values by mapping over each point on the grid
# add 1 onto each [x, y] pair to account for the constant term b0

# my original solution
zs = np.array([[Yhat(np.concatenate(([1], ii))) for ii in row] for row in grid])

# this also works
# xseval = [[np.concatenate(([1], ii)) for ii in row] for row in grid]
# zs = np.apply_along_axis(Yhat, 2, xseval)  # 2 is the 3rd axis

# this also works pt2
# xseval = np.ones((50, 50, 3))
# xseval[:, :, 1:] = grid
# zs = np.apply_along_axis(Yhat, 2, xseval)  # 2 is the 3rd axis

# xseval = np.concatenate(([1], grid))  # doesn't work - broadcasting only understands simple arithmetic operations:
# print(np.shape(np.array([1]) +  grid))  # this gives [i+1, j+1] for each point
# np.apply_along_axis(np.concatenate(([1], #)), 2, grid)  # something like Mathematica's slot???

ax.scatter(Xmat[:, 1], Xmat[:, 2], Y, color='red')
ax.plot_surface(xs, ys, zs, alpha=0.5)

plt.title('2D regression')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')

# multiline text that is compatible with LaTeX:
# https://stackoverflow.com/a/37930579/7799311
params = f'$R^2 = {round(Rsq, 3)}$'
ax.text(0, 0, 450, params, bbox={'facecolor': 'white'})

plt.show()
