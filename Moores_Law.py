import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import re

# load the data and turn into a numpy array
data_file = '/home/jkrys/Desktop/Programming/machine_learning_examples/linear_regression_class/moore.csv'
fo = open(data_file)

content = list(csv.reader(fo, delimiter='\t'))

# load only the 2nd and 3rd column (# of transistors, year)
# strip the data of unwanted stuff
# .+ means one or more occurrences of '.' (any character apart from '\n')
# [^0-9] means not(digits from 0 to 9)
pattern = "\[.+\]|[^\d]"
content = np.array([[int(re.sub(pattern, "", ii)) for ii in row[1:3]] for row in content])

Num, Year = content[:, 0], content[:, 1]

fo.close()

def get_lin_reg(X, Y):
    """
    Get the linear regression model of a data set (x, y).
    Returns coefficients (a, b, R^2), where a and b are coeffs of the linear fit.
    """

    # implement the equations for a and b which minimise the residual error
    denom = np.mean(X**2) - np.mean(X)**2
    a = (np.mean(X*Y) - np.mean(X)*np.mean(Y))/denom
    b = (np.mean(Y)*np.mean(X**2) - np.mean(X)*np.mean(X*Y))/denom

    def Yhat(x):
        return a * x + b  # best-fit line

    # get the R^2 value
    SSres = np.sum((Y - Yhat(X)) ** 2)
    SStot = np.sum((Y - np.mean(Y)) ** 2)
    Rsq = 1 - SSres/SStot

    return a, b, Rsq

a, b, Rsq = get_lin_reg(Year, np.log(Num))

time_to_double = np.log(2)/a

print(f'(a, b, R^2): {a, b, Rsq}')
print(f'Time to double transistor count is {time_to_double}')
# plot the data
# enable LaTeX in figure text
plt.rcParams.update({
    "text.usetex": True
})

plt.scatter(Year, Num, marker='.', color='blue')
plt.plot(Year, np.exp(a*Year + b), '-r')  # make this exponential because of the line below
plt.yscale('log')  # make the y-axis logarithmic

plt.title('Moore\'s Law')
plt.xlabel('Year')
plt.ylabel('Number of transistors')

# multiline text that is compatible with LaTeX:
# https://stackoverflow.com/a/37930579/7799311
params = f'$a = {round(a, 2)}$\n' + f'$b = {round(b, 2)}$\n' + f'$R^2 = {round(Rsq, 2)}$'
plt.text(1970, 10**9, params, bbox={'facecolor': 'blue', 'alpha': 0.2})

plt.show()
