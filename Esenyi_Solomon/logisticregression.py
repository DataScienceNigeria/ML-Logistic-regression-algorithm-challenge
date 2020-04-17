import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


class LogisticRegression:
    """
    Logistic regression implementation in Python

    fit(X, y) - data to fit to the model
    predict(x) - predict the classification of y from x
    predict_proba(x, threshold) - predict the value of y from x
    """

    def __init__(self, lr=0.01, num_iter=100000):
        self.lr = lr
        self.num_iter = num_iter

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

    def predict_proba(self, X):
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold


X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1, random_state=14)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# accuracy
print("accuracy:", accuracy_score(y_test, preds))
# f1 score
print("f1 score:", f1_score(y_test, preds))
