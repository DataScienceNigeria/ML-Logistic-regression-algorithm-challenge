import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Generating a logistic regression model
class LogisticRegression():

    # A constructor to initalize all my values
    def __init__(self, epochs=1000, alpha=0.01, n=0, m=0, lambda_=0.001, theta=0, theta0=0, gradients={}, parameters={}):
        self.epochs = epochs
        self.alpha = alpha
        self.lambda_ = lambda_
        self.X = X
        self.y = y
        self.m = m # rows
        self.n = n # columns
        self.theta = theta # weights
        self.theta0 = theta0 # bias
        self.gradients = gradients
        self.parameters = parameters

    def minMaxNormalization(self, X):
        # using the min-max normalization technique to normalise our input feature
        # to make sure gradient descent doesn't take too much time to converge
        return (X - X.min()) / (X.max() - X.min())

    # Randomly initialize random values for theta (weights) and theta0 (bias) to 
    # speed up gradient descent and let our logistic regression model perform
    # interesting non-linear outputs
    def initialize_theta(self, n):
        self.theta = np.random.randn(n, 1) 
        self.theta0 = np.random.randn(1)
    
    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-Z))

    # Here the forward propagation is going on here
    def forwardPropagation(self, X_train, y_train):
        z = np.dot(self.theta.T, X_train)
        h = self.sigmoid(z) # varies between 0 and 1

        reg = self.lambda_  * np.sum(self.theta**2) # regularization here
        cost = - y_train * np.log(h) - (1 - y_train) * np.log(1 - h) # cost function 
        cost = (np.sum(cost) + reg)/self.m 

        # backward propagation
        derivative_weights = np.dot(X_train, ((h-y_train).T)) / self.m
        derivative_bias = np.sum(h-y_train) / self.m
        self.gradients = {'derivative_weights' : derivative_weights,
                    'derivative_bias' : derivative_bias}
        return cost

    # updating our parameters {weights and bias}
    def update_weights(self, X_train, y_train):
        cost_list = []
        for i in range(self.epochs):
        	# computing the forward propagation
            cost = self.forwardPropagation(X_train, y_train)
            cost_list.append(cost)

            # Here we are updating our theta (weights) and theta0 (bias values)
            self.theta = (self.theta * (1 - (self.alpha * self.lambda_)/self.m))  - (self.alpha * self.gradients['derivative_weights'] + ((self.lambda_) * np.sum(self.theta))/self.m)
            self.theta0 = self.theta0 - self.alpha * self.gradients['derivative_bias']

            if i % 10000 == 0:
            	print('Epochs ==>> ',i)

        self.parameters = {'weight': self.theta, 'bias' : self.theta0}
        plt.plot(range(len(cost_list)), cost_list)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost function J(theta)')
        plt.show()

    # Training the model
    def fit(self, X_train, y_train):
        X_train = X_train.T
        y_train = y_train.T
        self.m = X_train.shape[0]
        self.n = X_train.shape[1]
        self.initialize_theta(self.m) # randomly initialize our theta values
        X_train = self.minMaxNormalization(X_train) # normalizing our input features
        self.update_weights(X_train, y_train) # updating our weights parameters 

    # Make predictions for us
    def predict(self, X_test):
        self.theta = self.parameters['weight'] # optimium weight
        self.theta0 = self.parameters['bias'] # optimum bias
        X_test = X_test.T
        z = self.sigmoid(np.dot(self.theta.T, X_test) + self.theta0)
        y_prediction = np.zeros((1, X_test.shape[1])) # number of samples
        for i in range(z.shape[1]):
            if z[0, i] < 0.5:
                y_prediction[0, i] = 0
            else:
                y_prediction[0, i] = 1
        return y_prediction[0]


# loading the data
breast_cancer = load_breast_cancer()

# intializing our feature matrix and target vectors
X = breast_cancer.data
y = breast_cancer.target

# checking the shape of our models
print(np.shape(X))
print(np.shape(y))

# spliting our data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# checking the shape to validate what was done
print('Checking the shapes...')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print('')

# calling our logistic regression method. I also integrated the regularization feature to help
# prevent our model from overfitting. So penalties are been put on our theta (weight) values
model = LogisticRegression(epochs=100000, lambda_=0.001, alpha=0.01)

# Here is where the training happens
print('Starting the training process...... [100,000 Iterations, alpha=0.01, lambda_=0.001]')
model.fit(X_train, y_train)
print('The training process is complete')

# made some predictions and stored the result here
result = model.predict(X_test)

# The accuracy of the model on our test dataset
print('\n\nAccuracy on test data: {:.2f}% \n\n'.format(sum(result == y_test) / len(y_test) * 100))