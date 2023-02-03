# demonstration of L1 regularization
#
# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python

import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50  # this will be a 'fat' matrix, i.e. many features

X = (np.random.random((N, D)) - 0.5)*10

# set some arbitrary w so that we can calculate Y
# we create it such that only the first 3 weights are relevant, the rest are 0 and are superfluous
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))
true_w = true_w[:, np.newaxis]

# real data set will have some variations around X*w
Y = X.dot(true_w) + np.random.default_rng().normal(0, 1, (N, 1))*0.5

eta = 10**(-3)

fig, (ax1, ax2) = plt.subplots(1, 2)
for l1 in range(0, 60+1, 15):
    w = np.random.default_rng().normal(0, 1 / np.sqrt(D), (D, 1))  # initialise w
    costs = []
    for i in range(50):
        Yhat = X.dot(w)  # model prediction
        delta = Yhat - Y
        w = w - eta*(np.transpose(X).dot(delta) + l1*np.sign(w))

        mse = np.sum(delta**2)
        costs.append(mse)

    ax1.plot(costs, label=f'l1: {l1}')
    ax2.plot(w, label=f'l1: {l1}', alpha=0.4)

params = f'$\eta = {eta}$\n'

ax1.set_xlabel('Iteration')
ax1.set_ylabel('MSE')
ax1.set_title('Mean square error as in the gradient descent algorithm')
ax1.legend()
plt.text(0.5, 0.8, params, fontsize=12, transform=ax1.transAxes)

ax2.plot(true_w, label='true w', linewidth=2)
ax2.set_xlabel('i')
ax2.set_ylabel('w_i')
ax2.set_title('True and calculated weights')
ax2.legend()
plt.show()

# as we can see, the L1 regularisation has coped with the unnecessary weights and returned only the first 3 as the
# relevant ones
# finding the right l1 value is a bit of trial-and-error
