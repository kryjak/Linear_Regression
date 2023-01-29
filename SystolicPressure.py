import numpy as np
import matplotlib.pyplot as plt
import xlrd
import pandas as pd

book = xlrd.open_workbook('/home/jkrys/Desktop/Programming/machine_learning_examples/linear_regression_class/mlr02.xls')
df = pd.read_excel(book)

data = df.to_numpy()

print(f'N: {np.shape(data)[0]}, D: {np.shape(data)[1] - 1}')
Y, Xmat = data[:, 0], data[:, 1:]

Xmat = np.c_[np.ones(Xmat.shape[0]), Xmat]  # 0.9768471041502091

# Adding a new parameter will have R^2>=0, but almost definitely R^2
# This is because there's always some correlation, no matter how small
# The worst we can predict by following linear regression is the mean, i.e. R^2
# So adding a new parameter will NEVER decrease R^2!
# Xmat = np.c_[np.ones(Xmat.shape[0]), Xmat, np.random.randint(-99999999,-999999, Xmat.shape[0])]

# Y = Y + np.random.randint(1000, 1000000, Xmat.shape[0])  # decreases R^2, because we add randomness to the output

w = np.linalg.solve(np.matmul(np.transpose(Xmat), Xmat), np.matmul(np.transpose(Xmat), Y))

def Yhat(x):
    return np.matmul(np.transpose(w), x)

SSres = np.sum((Y - Yhat(np.transpose(Xmat))) ** 2)
SStot = np.sum((Y - np.mean(Y)) ** 2)
Rsq = 1 - SSres/SStot

print(f'R^2 is {Rsq}')

### plotting
ax = plt.subplot(projection='3d')

xs = np.linspace(Xmat[:, 1].min(), Xmat[:, 1].max())
ys = np.linspace(Xmat[:, 2].min(), Xmat[:, 2].max())
xs, ys = np.meshgrid(xs, ys)

grid = np.reshape(np.transpose([xs, ys]), (50, 50, 2))

zs = np.array([[Yhat(np.concatenate(([1], ii))) for ii in row] for row in grid])

ax.scatter(Xmat[:, 1], Xmat[:, 2], Y, color='red')
ax.plot_surface(xs, ys, zs, alpha=0.5)

plt.title('Predicting blood pressure based on age and weight')
plt.xlabel('Age (yrs)')
plt.ylabel('Weight (lbs)')
ax.set_zlabel('Blood pressure')

params = f'$R^2 = {round(Rsq, 3)}$'
ax.text(45, 170, 160, params, bbox={'facecolor': 'white'})

plt.show()
