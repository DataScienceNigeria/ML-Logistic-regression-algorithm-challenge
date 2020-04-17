#
# This package contains functions which perform gradient descent operations
# which would be used throughout the module
#
from logreg.utilities import in_random_order, safe, step, negate, negate_all
import numpy as np

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    """
    Performs gradient descent across a function and its derivative
    Argument:
        target_fn (function) : A function whose error we want to minimize
        gradient_fn (function) : The derivative of our target function which we are going to minimize
        x (list) : A list of input
        y (list) : A ilst of expected outputs
        theta_0 (list): A list of initial guess of the gradients
        alpha (int) : A regularisation parameter

    Returns:
        theta (list): the coefficients of the linear logistic function 

    """
    data = zip(x, y)
    theta = theta_0  # initial guess
    alpha = alpha_0  # initial step size
    min_theta, min_value = None, float("inf") # the minimum so far
    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )
        
        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9
            # and take a gradient step for each of the data points
            
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = np.subtract(theta, np.multiply(alpha, gradient_i))
            
    return min_theta

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    theta = theta_0        # set theta to initial value
    target_fn = safe(target_fn)         # safe version of target_fn
    value = target_fn(theta)           # value we're minimizing
    
    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
            for step_size in step_sizes]
        
        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        
        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
    negate_all(gradient_fn),
    theta_0,
    tolerance)