# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:24:20 2020

@author: Professor
"""

import numpy as np
# next i define the sigmoid function. of course the sigmoid converts my values to probabilities thereby placing my values between 0 and 1 needed for (basic) classification.
def sigmoid(victor):
    return 1/(1 + np.exp(-victor))
# the cost function is next!! this quantifies the difference between the true and predicted values. this guy gives me the ability to do hyper parameter optimization as well as scoring.
def compute_cost(X, y, theta):
    # as you know,MAE requires the formular (1/m)*epsilon(y - y1)
    m = len(y)
    # simple matrix multiplication of x and theta
    h = sigmoid(X @ theta)
    # im simply defining the epsilon 10 raised to(-5)
    epsilon= 1e-5
    cost= (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost
# lets come over to Gradient descent, of course our model must follow the gradient of the cost function in order to reduce error in the next prediction as it goes down the array.
def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))
    
    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
        cost_history[i] = compute_cost(X, y, params)
    return(cost_history, params)
    # lastly lets define the predict function.
#of course i must not forget to round it of to both extremes b4 i turn it to a predict proba...lol.
def predict(X, params):
    return np.round(sigmoid(X @ params))
#DONE!!! TEST THE REGRESSION FUNCTION, IT WORKS PERFECTLY FINE. LEMME TEST IT.