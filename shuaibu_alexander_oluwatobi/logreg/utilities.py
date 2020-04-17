#
# This package contains random utility functions 
# which would be used throughout the module
#

import random

def negate(f):
    """
        Returns a function that for any input x returns -f(x)
        Arguments:
            f : A function f(x)

        Returns:
            -f : the negative of a function -f(x)

    """
    return lambda *args, **kwargs: -f(*args, **kwargs)


def negate_all(f):
    """
    the same when f returns a list of numbers
    Arguments:
        f : A function f(x) which returns a list as an output
    
    Returns : 
        -f : A function -f(x) which returns a list as an output
    """
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]


def in_random_order(data):
    """
    generator that returns the elements of data in random order
    Arguments:
        data(list) : An input List which we want to return on a random order
        
    Returns :
        data(list) : returns the items of the data in a random order
    """
    indexes = [i for i, _ in enumerate(data)] # create a list of indexes
    random.shuffle(indexes)
    # shuffle them
    for i in indexes:
    # return the data in that order
        yield data[i]

def safe(f):
    """
    Return a new function that's the same as f,
    except that it outputs infinity whenever f produces an error,
    this ensures that the function never crashes
    """
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')  
    return safe_f

def step(v, direction, step_size):
    """
    move step_size in the direction from v
    for moving in gradient descent
    """
    return [v_i + step_size * direction_i
    for v_i, direction_i in zip(v, direction)]
