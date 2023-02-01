# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

N = 100
X = np.linspace(0, 6 * np.pi, N)
Y = np.sin(X)

# in this file, we'll use polynomial regressions, so our inputs are x^0, x^1, ..., x^n

def make_poly(X, deg):
    """create the input matrix, where the first column is 1, followed by x^i terms of polynomial regression"""
    n = len(X)
    data = [np.ones(n)]
    for d in range(deg):
        data.append(X**(d+1))
    return np.vstack(data).T

print(np.shape(make_poly(X, 2)))  # (100xdeg+1) matrix

def fit(X, Y):
    """get the vector w of weights"""
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))


def fit_and_display(X, Y, sample, deg):
    """Solve the polynomial regression model based on samples from the full data set."""
    # get the training set
    sda
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    plt.plot(X, Y, label='Sin curve')
    plt.scatter(Xtrain, Ytrain, label='Training points')
    # plt.title('Sample points drawn from a sin curve')
    # plt.show()

    # get the weights of the polynomial model using the training sample of (x, y) coordinates
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)

    # use these weights to construct a polynomial which is meant to model full data
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)

    # display the model
    plt.plot(X, Y_hat, label='Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Polynomial regression model:\n samples: {sample}, degree: {deg}.')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: f'{int(val/np.pi)}$\pi$' if val != 0 else '0'
    ))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    plt.show()


def get_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d) / len(d)


def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    # get the training set
    N = len(X)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    # get the test set
    test_idx = [idx for idx in range(N) if idx not in train_idx]
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []
    for deg in range(max_deg+1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)  # weights based on the training set
        Yhat_train = Xtrain_poly.dot(w)  # y-values for these sample points
        mse_train = get_mse(Ytrain, Yhat_train)  # Mean squared error of the training data

        Xtest_poly = make_poly(Xtest, deg)  # weights based on the test set
        Yhat_test = Xtest_poly.dot(w)  # y-values for these sample points
        mse_test = get_mse(Ytest, Yhat_test)  # Mean squared error of the test data

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(range(max_deg+1), mse_trains, label="train MSE")
    plt.plot(range(max_deg+1), mse_tests, label="test MSE")
    plt.xlabel('Polynomial degree')
    plt.ylabel('Mean square error')
    plt.title(f'Model mean squared error: {sample} samples.')
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label="train mse")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# in this case, samples are selecting from data points lying on a sin curve
if __name__ == "__main__":
    # make up some data and plot it
    N = 100
    X = np.linspace(0, 6*np.pi, N)
    Y = np.sin(X)
    sample = 10

    plt.plot(X, Y)
    # plt.show()

    train_idx = np.random.choice(len(X), sample)  # choose a random sample of points from the full data set
    for deg in (5, 6, 7, 8,  9):
        fit_and_display(X, Y, sample, deg)
    plot_train_vs_test_curves(X, Y)

# we can see that the effectiveness of the model depends a lot on the samples that were drawn from the data set
# the model goes crazy in areas where there are no sample points
# from this reason, it is important to collect a lot of data so that we can get a representative sample

