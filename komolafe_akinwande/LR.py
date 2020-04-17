import numpy as np

#Lets implement the following

#Activation (sigmoid function)
def sigmoid(x):
    ''' A logit of a record or an event. It returns a value between 0 and 1'''
    return (1 / (1 + np.exp(-x)))

#using multiple linear models 
#differential of sigmoid function is 
def sigmoid_prime(x):
    ''' Returns the differential of the sigmoid function. When using mutiple linear models or perceptron'''
    return sigmoid(x)* (1-sigmoid(x))


#Ouptut(prediction function)
def prediction(features, weight,bias):
    '''Returns the probability of an event, which is always between 0 and 1.
       Parameters
       ----------
    features : the independent variables or predictors
    weight: Random initialised weight
    bias: Random initialised weight'''
    return sigmoid(np.dot(features,weight)+ bias)


#error (log-loss function) for a single perceptron
def error_formula(y,output):
    '''Returns the value of the error, Which just takes the uncertainty of the predictions based on 
       how much it varies from the actual label.
       
       Parameters
       ----------
    y : The dependent variables or the target/label
    output: The prediction, which is between 0 and 1
    '''
    return -y*np.log(output) - (1-y)*np.log(1-output)

#error (log-loss function) for multiple perceptron
def error_fucn(x,y,output):
    differential = sigmoid_prime(x)
    error = y - output
    return error*differential


#gradient descent step
def update_weights(x,y, weight,bias, learnrate):
    '''Returns the value of the updated weight and bias. 
       
       Parameters
       ----------
    x : The independent variables
    y : The dependent variables or the target/label
    weight: Random initialised weight
    bias: Random initialised weight
    learnrate : proportional steps to take for every iteration
    '''
    pred = prediction(x,weight,bias)
    error = y - pred
    weight += learnrate * error * x
    bias += learnrate * error
    return weight,bias


#seed random numbers to make calculation deterministic
np.random.seed(45)
class LogisticRegression:
    '''Compute a Logistic Regression model for a list of hyperparameters like epochs and learnrate'''
    #Class Constructor
    def __init__( self, epochs,learnrate):
        self.epochs = epochs
        self.learnrate = learnrate
        
    def fit(self, features, targets):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        features : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        targets : array-like of shape (n_samples,)
            Target vector relative to X.
        epoch  : The number of times the model should iterate in order to update the weight and the bias
        learnrate : proportional steps to take for every iteration
       
        Returns
        -------
        Updated Weight and Bias"""
    
        #assign a random weight and bias zero
        n_records, n_features = features.shape
        weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
        bias = 0
        for iter in range(self.epochs):
            for x,y in zip(features,targets):
                output = prediction(x,weights,bias)
                error = y - output
                weights, bias = update_weights(x,y,weights,bias,self.learnrate)
        return (weights,bias)
                
    def predict(self,features,targets,weights,bias):
        """
        Make predictions using the updated weight and bias.
        Parameters
        ----------
        features : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        targets : array-like of shape (n_samples,)
            Target vector relative to X.
        weight: The updated weight
        bias: The updated bias
       
        Returns
        -------
        Predictions in form of probabilities"""
        
        errors = []
        last_loss = None
        for iter in range(self.epochs):
        #print out log loss for training dataset
            out = prediction(features,weights,bias)
            loss = np.mean(error_formula(targets,out))
            errors.append(loss)
            if iter % (self.epochs/10) == 0:
                print('\n==============Epoch', iter, "========")
            if last_loss and last_loss < loss:
                print('Training loss:', loss, " WARNING ---  Loss is increasing")
            else:
                print('Training loss', loss)
            last_loss  = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions==targets)
            print("Accuracy: ", accuracy)
        return out