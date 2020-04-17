#
# This package contains functions which perform statistical operations
# which would be used throughout the module
#

import math
from logreg.vectors import sum_of_squares

def mean(x):
    """
    Returns the mean of an element
    Argument:
        x(list) : takes in a list as an argument
    Returns:
        int :  the mean of the Vector(list)
    """
    return sum(x) / len(x)

def standard_deviation(x):
    """
    Returns the Standard deviation of a vector
    Argument:
        x(list) : takes in a list as an argument
    Returns:
        int :  the std of the Vector(list)
    """
    return math.sqrt(variance(x))

def de_mean(x):
    """
    Translate x by subtracting its mean (so the result has mean 0)
    Argument:
        x(list) : takes in a list as an argument
    Returns:
        list : returns the difference between each element and its mean
    """
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    """
    Assumes x has at least two elements, and calculates the variance between the elements
    Argument:
        x(list) : takes in a list as an argument
    Returns:
        int : The variance of the list/vectors
    """
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def get_column(A, j):
    """
    return only a particular column in a list
    """
    return [A_i[j]
    for A_i in A] # jth element of ro

def shape(A):
    """
    returns the shape of a multidimentional array
    """
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def make_matrix(num_rows, num_cols, entry_fn):
    """
    returns a num_rows x num_cols matrix
    whose (i,j)th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)       # given i, create a list
        for j in range(num_cols)] # [entry_fn(i, 0), ... ]
        for i in range(num_rows)] # create one list for each i

def scale(data_matrix):
    """returns the means and standard deviations of each column"""
    _, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix,j))
                for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix,j))
                for j in range(num_cols)]
    return means, stdevs

def rescale(data_matrix, means = 0, stdevs = 0):
    """
    rescales the input data so that each column
    has mean 0 and standard deviation 1
    leaves alone columns with no deviation (i.e columns which have the same value through out)
    """
    if not (means and stdevs):
        means, stdevs = scale(data_matrix)
    
    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]
        
    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)