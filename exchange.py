import numpy as np
from sklearn.datasets import load_breast_cancer
data  = load_breast_cancer()

X = data.data[:, :2]
y = (data.target != 0) * 1
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=10, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def intercepts(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.intercepts(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(i % 100 == 0):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                print(f'loss: {self.loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.intercepts(X)
    
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
model = LogisticRegression(lr=0.1, num_iter=30000)
model.fit(X, y)
preds = model.predict(X, threshold=10)
# accuracy
accuracy = print(preds)