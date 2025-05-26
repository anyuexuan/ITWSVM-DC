import numpy as np


def Kernel(A, B=None, kernel='rbf', gamma=0.1, q=0.1, degree=3):
    kernel = str(kernel).lower()
    A = np.array(A, np.float)
    if len(A.shape) == 1:
        A = np.expand_dims(A, 1)
    if B is None:
        B = A.copy()
    B = np.array(B, np.float)
    if len(B.shape) == 1:
        B = np.expand_dims(B, 1)
    if kernel == 'linear':
        my_kernel = A.dot(B.T)
    elif kernel == 'polynomial' or kernel == 'poly':
        my_kernel = (gamma * A.dot(B.T) + q) ** degree
    elif kernel == 'sigmoid':
        my_kernel = np.tanh(gamma * A.dot(B.T) - q)
    elif kernel == 'rbf':
        rA = np.sum(np.square(A), 1, keepdims=True)
        rB = np.sum(np.square(B), 1, keepdims=True)
        sq_dists = rA - 2 * A.dot(B.T) + np.transpose(rB)  # x^2-2*x*y+y^2
        my_kernel = np.exp(-gamma * np.abs(sq_dists))
    else:
        print('kernel error!')
        return
    return my_kernel
