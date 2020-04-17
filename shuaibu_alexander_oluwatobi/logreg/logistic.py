#
# This package contains functions which perform logistic operations
# which would be used throughout the module
#

import math
from numpy import dot
from functools import reduce
from logreg.vectors import vector_add

def logistic(x):
    """
    A logistic function with x as an input

    Argument:
        x : an integer
    Returns:
        an exponential function with x as a variable
    """
    return 1.0 / (1 + math.exp(-x))


def logistic_prime(x):
    """
    The derivaive of the logistic equation

    Argument:
        x : an integer
    Returns:
        the derivative of an exponential function with x as a variable
    """
    return logistic(x) * (1 - logistic(x))


def logistic_log_likelihood_i(x_i, y_i, beta):
    """
    The log likelyhood of  an element
    """
    if y_i == 1:
        return math.log(logistic(dot(x_i, beta)))
    else:
        return math.log(1 - logistic(dot(x_i, beta)))


def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
            for x_i, y_i in zip(x, y))


def logistic_log_partial_ij(x_i, y_i, beta, j):
    """
    here i is the index of the data point, j the index of the derivative
    """
    return (y_i - logistic(dot(x_i, beta))) * x_i[j]


def logistic_log_gradient_i(x_i, y_i, beta):
    """
    the gradient of the log likelihood
    corresponding to the ith data point
    which would be minimized using gradient descent
    """
    return [logistic_log_partial_ij(x_i, y_i, beta, j)
            for j, _ in enumerate(beta)]



def logistic_log_gradient(x, y, beta):
    """
    the gradient of the log likelihood
    of the entore dataset
    """
    return reduce(vector_add,[logistic_log_gradient_i(x_i, y_i, beta)
            for x_i, y_i in zip(x,y)])
