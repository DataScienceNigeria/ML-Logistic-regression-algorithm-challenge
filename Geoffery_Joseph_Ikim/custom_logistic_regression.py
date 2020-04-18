import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class CustomLogisticRegression:
    """ Implements Logistic Regression with support for multiple classes
   
    
    """

    def __init__(self, alpha=0.01, iterations=1000):
        """ Constructor definition
        
        Keyword Arguments:
            alpha {float} -- Learning rate (default: {0.01})
            iterations {int} -- Number of iterations (default: {1000})
        """
        self._cost_list = list()  # Stores Cost after optimization
        self.coef = list()  # Stores all theta (parameters) values
        self._n_iters = iterations      
        self._alpha = alpha

        self.x = []
        self.y = []
    
    def _sigmoid(self, z):
        """ This is the Sigmoid function 

        Reduces result of linear functions to a probability
        
        Arguments:
            z {ndarray} -- Linear function or result thereof
        
        Returns:
            ndarray -- A (or list of) probability(ies) values
        """

        return 1 / (1 + np.exp(-z))
    
    def _cost(self, x, y, theta):
        """ Calculates the cost (difference between the expected values and observed values )
        
        Arguments:
            x {ndarray} -- Features (Independent variable)
            y {ndarray} -- Target (Dependent variables)
            theta {ndarray} -- Parameters 
        
        Returns:
            float -- sum of difference between the hypothesis and target
        """
        m = x.shape[0]  # number of training examples (rows of X)
        h = self._sigmoid(x @ theta)
        e = 1e-9  # small number added to prevent division by zero in the log function
        return ((-y.T @ np.log(h + e)) - ((1 - y).T @ np.log(1 - h + e))) / m

    def _gradient(self, x, y, theta):
        """ Calculates the gradient
        
        Arguments:
            x {ndarray} -- Features (Independent variable)
            y {ndarray} -- Target (Dependent variables)
            theta {ndarray} -- Parameters 
        
        Returns:
            float -- derivative of the cost function
        """
        h = self._sigmoid(x @ theta)
        return (1 / len(y)) * (x.T @ (h - y))

    def _gradient_descent(self, x, y, alpha):
        """ Implements gradient Descent to minimize (optimize) the cost function 
        
        Arguments:
            x {ndarray} -- Features (Independent variable)
            y {ndarray} -- Target (Dependent variables)
            alpha {float} -- Learning rate 
        
        Returns:
            ndarray -- optimized parameters 
        """
        
        theta = np.zeros(x.shape[1])
        cost = []

        for _ in range(self._n_iters):
        
            theta -= alpha * self._gradient(x, y, theta)  # simultaneously update all parameters
            cost.append(self._cost(x, y, theta))
            
        self._cost_list.append(cost)
        return theta

    def fit(self, x, y):
        """ Trains the model (tries to get optimal parameters that'll reduce cost)
        
        Arguments:
            x {ndarray} -- Features (Independent variable)
            y {ndarray} -- Target (Dependent variables)
            
        """
        
        self.y = y
        self.x = np.insert(x, 0, 1, axis=1)
        labels = np.unique(self.y)  # List of classes in the target variable
        
        for i in labels:
            """ Iterates through the classes in the target 

            this is the implementation of ONE VS. ALL classification
            where the current label is set to 1 and the rest of the labels are set to 0
            """
            label = (self.y == i).astype(int)  # sets values of the current class to one, every other class to zero 
            self.coef.append(self._gradient_descent(self.x, label, self._alpha))

    def predict(self, x):
        """ Predict y Given X
        
        Arguments:
            x {1d ndarray} -- values for predicting y
        
        Returns:
            int -- label corresponding to the max probability 
        """
        x = np.insert(x, 0, 1)
        self.coef = np.array(self.coef)
        probabilities = self._sigmoid(x @ self.coef.T)  # returns a list of probabilities

        return probabilities.argmax()  # returns the index of the maximum probability value
   
    def plot_cost(self):
        """ Plots the cost over the course of optimization
        """
        
        for cost in self._cost_list:
            i = self._cost_list.index(cost) 
            ax = range(len(cost))
            plt.plot(ax, cost)
            plt.title(f"cost for {i}")
            plt.xlabel('iterations')
            plt.ylabel('cost')
            
            plt.show()
    
    def accuracy(self):
        """ Calculates the accuracy of the algorithm
        
        Returns:
            float -- Accuracy percentage 
        """
        y_pred = []
        x = np.delete(self.x, 0, 1)
        for i in x:
            
            y_pred.append(self.predict(i))
        
        return (float(np.sum(y_pred == self.y)) / float(len(self.y))) * 100


if __name__ == '__main__':
    # Test datasets 1

    # from scipy.io import loadmat
    # data = loadmat('digits.mat')
    # np.place(y, y == 10, 0)

    # Test datasets 2

    # data = pd.read_csv('logistic_regression_data.txt', header=None)
    # x = data.iloc[:,:2].values
    # y = data.iloc[:,2].values

    # Test datasets 3

    # from sklearn.datasets import load_iris
    # data = load_iris()
    # x = data['data']
    # y = data.target

    # Test datasets 4

    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    
    # Model initialization
    model = CustomLogisticRegression(alpha=0.01, iterations=1000)

    # model training
    model.fit(x, y)

    # model accuracy test
    print("accuracy", model.accuracy())
    print(model.predict(x[50]), y[50])

    # model cost plots
    model.plot_cost()