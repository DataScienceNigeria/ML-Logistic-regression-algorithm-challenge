#
# This package contains functions which perform vector operations
# which would be used throughout the module
#

from numpy import dot,add

def vector_add(v, w):
    """
    adds corresponding elements
    Arguments:
        v (list) : A list containing vectors
        w (list) : A list containing vectors
    Returns:
        The sum of two vectors
    """
    return add(v,w)


def sum_of_squares(v):
    """
    Returns the Sum of squares of two vectors
    Arguments:
        v (list) : A list containing vectors
    Returns:
        the dot product of two vectors
    """
    return dot(v,v)


