"""
Quick numpy tutorial based on:
https://www.youtube.com/watch?v=QUT1VHiLmmI&ab_channel=freeCodeCamp.org
"""

"""
Numpy is a multidimensional array library.
It's much faster than the normal lists in Python:
1. Numpy uses fixed types and so we can save memory because we don't store as much information.
   Moreover, when iterating over a list in numpy, we don't have to do type checking on each element.   
2. Numpy uses contiguous memory - consecutive memory blocks are assigned to a process/file.
   https://www.geeksforgeeks.org/difference-between-contiguous-and-noncontiguous-memory-allocation/
   SIMD vector processing
   Effective cache utilisation
   
https://numpy.org/doc/stable/reference/
"""

import numpy as np
a = np.array([1, 2, 3])  # defining an array
print(a)

b = np.array([1, 2.0, 3])
b2 = [1, 2.0, 3]
print(b)  # notice that all three numbers are printed as floats - fixed types in numpy!!!
print(b2)  # but normal lists allow mixed types

c = np.array([[1, 2, 3], [3, 4, 5]])  # a 2x3 matrix
print(c)
# c2 = np.array([[1, [9, 8, 7], 3], [3, 4, 5]])  # this is not allowed

print("BASIC OPERATIONS".center(50, "-"))
# https://numpy.org/doc/stable/reference/arrays.ndarray.html

print('ndim:', a.ndim, b.ndim, c.ndim)  # essentially how many indices are necessary to specify a position
print('shape:', a.shape, b.shape, c.shape)  # equivalent of Dimensions in Mathematica
# print(list(map(np.shape, [a, b, c])))

print('dtype:', a.dtype, b.dtype, c.dtype)
# we can also specify the type:
d = np.array([1, 2, -128, 127], dtype='int8')
print('specified dtype:', d.dtype)
print(np.iinfo('int8'))
# the largest number we can represent with 1 byte is 127
# 1 byte can store 2^8 = 256 values - in numpy, they're taken as (-128, ..., 127)
# we can also get this information by using 'np.iinfo'

print('itemsize:', a.itemsize, b.itemsize, c.itemsize, d.itemsize)  # size of ONE array element in bytes
print('size:', a.size, b.size, c.size, d.size)  # total number of elements
print('size along an axis:', np.size(c, 0), np.size(c, 1))
print('nbytes:', a.nbytes, b.nbytes, c.nbytes)
# print('total size of c in bytes:', np.size(c)*c.itemsize)  # same thing

print("SLICING ETC.".center(50, "-"))
# slicing is the same as for normal lists
a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
print(a[1, 5])
print(a[:, 2:5])

print("INITIALISING ARRAYS".center(50, "-"))
# https://numpy.org/doc/stable/reference/routines.array-creation.html
# create a constant array with 0, 1 or any other number:
print(np.zeros((2, 3)))
print(np.ones((2, 3, 4), 'int'))  # can also specify data type
print(np.full((2, 2), 99.1, 'float'))  # try specifying 'int' here
print(np.full_like(a, 3))  # create an array with the same dimensions and type as some other array
# there's also np.zeros_like and np.ones_like
print(np.identity(3, 'int'))  # 3x3 identity matrix
print(np.eye(3, 2, k=-1, dtype='int'))  # like identity, but we can specify the k-th diagonal
print(np.diagonal(a, 4))

print("-".center(10, "-"))
x = np.array([[1, 2], [3, 4]])
print(x)
print(np.repeat(x, 2))  # if axis is not specified, take flattened input
print(np.repeat(x, 2, 0))  # along axis 0: (2, 2) -> (4, 2)
print(np.repeat(x, 2, 1))  # along axis 1: (2, 2) -> (2, 4)
print(np.repeat(x, [2, 3], 0))  # along axis 0: (2, 2) -> (2+3, 4)
print(np.repeat(x, [2, 3], 1))  # along axis 1: (2, 2) -> (2, 2+3)

print("-".center(10, "-"))
test = np.ones((5, 5), 'int')
test[1:-1, 1:-1] = np.zeros((3, 3))
test[2, 2] = 9
print(test)

print("-".center(10, "-"))
# remember that an assignment (=) in Python creates a pointer to the same location in memory, not a copy of an object
a = np.array([1, 2, 3])
b = a
b[0] = 99
print(f'{a}\n\b{b}')
# instead, create a copy:
a = np.array([1, 2, 3])
b = a.copy()
b[0] = 99
print(f'{a}\n\b{b}')

print("-".center(20, "-"))
# https://numpy.org/doc/stable/reference/random/legacy.html#functions-in-numpy-random
print(np.random.random(4)) # 1D array of random floats from the interval [0, 1)
print(np.random.rand(1, 2, 3))  # n-dim array of floats from [0, 1)
print(np.random.uniform(99, 99999, (2, 3)))  # n-dim array of floats from a [low, high)
print(np.random.randint(99, 999999, (2, 3)))  # n-dim array of ints from a [low, high)

print("MATHEMATICS".center(50, "-"))
a = np.array([1, 2, 3, 4])
b = np.array([0, 1, 0, 1])
mat1 = np.random.randint(1, 9, (2, 3))
mat2 = np.random.randint(1, 9, (3, 2))
mat3 = np.random.randint(1, 9, (2, 3))

# basic mathematical operations:
print(a + 2)
print(a ** b)
print(np.cos(a))
# and many more: https://numpy.org/doc/stable/reference/routines.math.html#

print("LINEAR ALGEBRA".center(20, "-"))
# linear algebra:
# https://numpy.org/doc/stable/reference/routines.linalg.html

# np.dot is a generic function that takes many meanings depending on the contex. See documentation
print('vdot:', np.vdot(a, b))  # dot product between two vectors
print('matmul:\n', np.matmul(mat1, mat2))  # matrix multiplication
print('inner:\n', np.inner(mat1, mat3))  # inner product
print('outer:\n', np.outer(mat1, mat3))  # outer product
print('determinant:\n', np.linalg.det(np.identity(100)))  # outer product

print("STATISTICS".center(20, "-"))
# https://numpy.org/doc/stable/reference/routines.statistics.html
stat = np.array([[1, 2, 3], [4, 5, 6]])
print(stat)

print('sum:', np.sum(stat))
print('sum0:', np.sum(stat, 0))  # can also specify the axis
print('sum1:', np.sum(stat, 1))

print('min:', np.min(stat))
print('max:', np.max(stat))
print('mean:', np.mean(stat))
print('median:', np.median(stat))

print("REORGANISING".center(20, "-"))
# https://numpy.org/doc/stable/reference/routines.array-manipulation.html
print('reshape:\n', stat.reshape((3, 2)))
print('reshape2:\n', stat.reshape((3, 2, 1)))

print('vstack:\n', np.vstack([stat, stat]))
print('hstack:\n', np.hstack([stat, stat]))

print("MISCELLANEOUS".center(20, "-"))
# data from a file can be loaded into an array with np.genfromtxt

print(f'stat:\n\b{stat}')
print('boolean slice:', stat[[[True, False, True], [False, True, False]]])  # we can slice an array with Boolean values

print('>2:', stat > 2)  # logical tests: boolean if item > 2
print('>2:', stat[stat > 2])  # use this to slice a list

print('any:', np.any((stat > 2) & (stat < 4)))
print('any:', np.any((stat > 2) & (stat < 4), 0))  # specify the axis
print('any:', np.any((stat > 2) & (stat < 4), 1))

print('all:', np.all(stat > 0))

print("-".center(20, "-"))
test = np.arange(1, 31).reshape(6, 5)
print(test)

print(test[2:4, :2])
print(test[[0, 1, 2, 3], [1, 2, 3, 4]])
print(test[[0, 4, 5], 3:])
