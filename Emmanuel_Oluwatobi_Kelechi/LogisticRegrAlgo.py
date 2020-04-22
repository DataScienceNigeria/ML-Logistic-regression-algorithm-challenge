import numpy as np
class LogisticRegression:
    """Python's implementation of the logistic regression algorithm from scratch with the numpy library."""
    def __init__(self):
        pass

    def predict(self, X_test, weights):
        """
        :param X_test: An array of an array of all independent variable features
        :param weights: An array of weights corresponding to each feature in the X_test array
        :return: Returns the predicted Y value or probability
        """
        #Checks if there is an intercept row in the features array
        if X_test.shape[1] != weights.shape[0]:
            intercept = np.ones((X_test.shape[0], 1))
            X_test = np.hstack((intercept, X_test))
        z = np.dot(X_test, weights)
        return 1/(1+np.exp(-z))

    def fit(self, X_train, Y_train, max_iter, learning_rate):
        """
        :param X_train:  Training features
        :param Y_train:  Y value or class of the training features
        :param max_iter: Maximum number of iterations for the gradient decent algorithm
        :param learning_rate: Learning rate
        :return: returns an array of weights for each features
        """
        for iter_i in range(max_iter):
            x_train = X_train
            #include intercept in the X_train features
            intercept = np.ones((x_train.shape[0], 1))
            x_train = np.hstack((intercept, x_train))
            weights = np.zeros(x_train.shape[1])
            y_pred = self.predict(x_train, weights)
            error = Y_train-y_pred
            gradient = np.dot(x_train.T, error)
            weights += learning_rate*gradient
        return weights

    def score(self, y_test, y_pred):
        """
        :param y_test:The test value of the predictions
        :param y_pred: The predicted values from the LogitModel
        :return: returns a score in percentage of how correct the predictions are
        """
        y_pred = [0 if i < 0.5 else 1 for i in y_pred]
        return 'Model score: ', 100*np.mean(y_test == y_pred)




#TRAINING THE MODEL
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], 3)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], 3)
total = np.vstack((x1, x2)).astype(np.float32)
y_val = np.hstack((np.zeros(3), np.ones(3)))
model = LogisticRegression()
weights = model.fit(total, y_val, 10, 0.01)
print(total.shape)
print(y_val.shape)


#TESTING THE MODEL
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], 3)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], 3)
total = np.vstack((x1, x2)).astype(np.float32)
int = np.ones((total.shape[0], 1))
y_pred = model.predict(total, weights)
#convert the predicted proabilities to binary value
y_pred = [1 if i > 0.5 else 0 for i in y_pred]
y_val = np.hstack((np.zeros(3), np.ones(3)))
print(model.score(y_val, y_pred))








