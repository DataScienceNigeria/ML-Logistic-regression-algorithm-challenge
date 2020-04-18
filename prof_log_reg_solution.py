# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:24:20 2020

@author: Professor
"""

import numpy as np
def sigmoid(victor):
    return 1/(1 + np.exp(-victor))
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon= 1e-5
    cost= (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost
def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))
    
    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
        cost_history[i] = compute_cost(X, y, params)
    return(cost_history, params)
    def predict(X, params):
        return np.round(sigmoid(X @ params))
#DONE!!! TEST THE REGRESSION FUNCTION, IT WORKS PERFECTLY FINE. LEMME TEST IT myself.