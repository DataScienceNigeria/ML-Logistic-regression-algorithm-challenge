#!/usr/bin/env python
# coding: utf-8

from math import exp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# creating the logistic model class
class logisticRegression():
    def __init__(self, iteration_no=300, L=0.001, b0 = 0,b1 = 0):
        self.iteration_no = iteration_no
        self.L = L
        self.b0 = b0
        self.b1 = b1
        
    def normalize(self, X):
        return X - X.mean()

    # Method to make predictions
    def predict_prob(self,X, b0, b1):
        return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])

    # Method to train the model
    def fit(self,X, Y):

        X = self.normalize(X)

        for epoch in range(self.iteration_no):
            y_pred = self.predict_prob(X, self.b0, self.b1)
            D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
            D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
            # Update b0 and b1
            self.b0 = self.b0 - self.L * D_b0
            self.b1 = self.b1 - self.L * D_b1

    def predict(self, x_test, b0, b1):
        v = self.predict_prob(x_test, b0, b1)
        vr = [1 if p>= 0.5 else 0 for p in v]
        return vr

# Training the model
Log = logisticRegression()
Log.fit(X_train, Y_train)

# Making predictions
X_test_norm = normalize(X_test)
y_pred = Log.predict(X_test_norm, b0, b1)

# The accuracy
print("accuracy:", accuracy_score(Y_test, y_pred))
# f1 score
print("f1 score:", f1_score(Y_test, y_pred))





