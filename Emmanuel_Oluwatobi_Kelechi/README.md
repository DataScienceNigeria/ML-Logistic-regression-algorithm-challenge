# Python's implementation of Multiple and Simple logistic regression algorithm from scratch using numpy

Contains a LogisticRegression class with methods predict, fit, score.
The model.predict method predicts some probability based on an array input X-val and weights and returns the predicted probabilities.
The model.score method outputs the accuracy of the LogisticRegression algorithm in percentage by accepting an array of the predicted values converted to their binary form and the real values.
The predicted values is converted to binary 0 or 1 based on the users threshold specs.
The model.fit method takes in the max_iter and learning rate parameters with the x_train and y_train array for the gradient descent algorithm.


