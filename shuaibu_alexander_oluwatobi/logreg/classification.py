import random
import numpy as np
from functools import partial
from sklearn import metrics
from logreg.gradient import maximize_batch
from logreg.statistics import rescale, scale
from logreg.logistic import logistic_log_likelihood, logistic_log_gradient, logistic



def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test

def make_prediction(x,y):
    random.seed(0)
    means, stdevs = scale(x)
    rescaled_x = rescale(x, means=means, stdevs=stdevs)
    x_train, y_train = rescaled_x, y
    # want to maximize log likelihood on the training data
    fn = partial(logistic_log_likelihood, x_train, y_train)
    gradient_fn = partial(logistic_log_gradient, x_train, y_train)
    # pick a random starting point
    beta_0 = [random.random() for _ in range(3)]
    # and maximize using gradient descent
    beta_hat = maximize_batch(fn, gradient_fn, beta_0)
    return beta_hat

class LogisticRegression:

    def __init__(self):
        # initialise the variables
        self.beta_hat = []
        self.stdevs = None
        self.means = None
        self.fitted = False
        self.train_shape = None

    def _rescale(self, x):
        # a private function to rescale the data using the saved means and stdev
        return rescale(x, means=self.means, stdevs=self.stdevs)

    def fit(self,x,y_train):
        ""
        # convert the input to a numpy array if not already
        x = np.array(x)
        # make sure that a multi dimentional array is passed in
        assert len(x.shape) !=1 , "Please Provide a multidimentional array"

        # create the coefficients i.e the constant c in y = mx+c which would have no coeficcients
        coefficients = np.full(x.shape[0],1).reshape(-1,1)
        # concatenate the coefficients with the input array
        x_train = np.concatenate((coefficients, x),axis=1)
        # save the train shape
        self.train_shape = x_train.shape
        # get the initial coeficients of the equation
        beta_0 = [random.random() for _ in range(x.shape[1]+1)]
        x_train = x_train.tolist()
        # scale the input data and store the key values
        self.means, self.stdevs = scale(x_train)
        rescaled_x_train = self._rescale(x_train)

        # fit the model and get the parameters
        fn = partial(logistic_log_likelihood, rescaled_x_train, y_train)
        gradient_fn = partial(logistic_log_gradient, rescaled_x_train, y_train)
        beta_hat = maximize_batch(fn, gradient_fn, beta_0)

        # save the parameters
        self.fitted = True
        self.beta_hat = beta_hat

        # return self for the kulture ;)
        return self

    def predict(self,x):
        """
        Predict using the just trained model
        """
        x = np.array(x)
        assert len(x.shape) !=1 , "Please Provide a multidimentional array"
        assert self.fitted == True, "Please fit the model first"

        # add the coefficients
        coefficients = np.full(x.shape[0],1).reshape(-1,1)
        # concatenate the coefficients with the input array
        x_test = np.concatenate((coefficients, x),axis=1)

        assert self.train_shape == x_test.shape, "Please provide a test size the same shape as the train shape"
        # scale the data with the saved parameters
        x_test = x_test.tolist()
        rescaled_x_test = self._rescale(x_test)

        predictions = [logistic(np.dot(self.beta_hat, x_i)) for x_i in rescaled_x_test]

        return predictions
    
    