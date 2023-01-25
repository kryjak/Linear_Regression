import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import get_cachedir

# load the data and turn into a numpy array
data_file = '/home/jkrys/Desktop/Programming/machine_learning_examples/linear_regression_class/data_1d.csv'
fo = open(data_file)

content = np.array(list(csv.reader(fo)), dtype='float')
X, Y = content[:, 0], content[:, 1]

fo.close()

# now implement the equations for a and b which minimise the error
denom = np.mean(X**2) - np.mean(X)**2
a = (np.mean(X*Y) - np.mean(X)*np.mean(Y))/denom
b = (np.mean(Y)*np.mean(X**2) - np.mean(X)*np.mean(X*Y))/denom
# X*Y for numpy arrays is the same as np.multiply(X, Y) - Hadamard (element-wise) multiplication
# https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy

def Yhat(x):
    return a*x + b  # best-fit line

SSres = np.sum((Y - Yhat(X)) ** 2)
SStot = np.sum((Y - np.mean(Y)) ** 2)
Rsq = 1 - SSres/SStot

print(f'R^2 is {Rsq}')

# plot the data
plt.scatter(X, Y)
plt.plot(X, Yhat(X))
print(get_cachedir())
plt.title('Linear regression')

plt.show()
