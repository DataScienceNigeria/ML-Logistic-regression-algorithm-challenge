#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

