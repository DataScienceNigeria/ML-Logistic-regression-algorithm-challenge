#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from SigmoidFunction import sigmoid
def cost(coefs,X,y,lmbda):
    m = len(y)
    y_1 = np.multiply(y,np.log(sigmoid(np.dot(X,coefs))))
    y_0 = np.multiply(1-y,np.log(1-sigmoid(np.dot(X,coefs))))
    return np.sum(y_1 + y_0) / (-m) + np.sum(coefs[1:]**2) * lmbda /(2*m)
def gradient(coefs,X,y,lmbda):
    m = len(y)
    error = sigmoid(np.dot(X,coefs)) - y
    grad_coefs = np.dot(X.T,error) / m + coefs * lmbda / m
    grad_coefs[0] = grad_coefs[0] - coefs[0] * lmbda / m
    return grad_coefs

