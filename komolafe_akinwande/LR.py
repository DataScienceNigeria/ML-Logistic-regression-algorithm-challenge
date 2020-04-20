import numpy as np

#seed random numbers to make calculation deterministic
np.random.seed(45)
class LogisticRegression:
    '''Compute a Logistic Regression model for a list of hyperparameters like epochs and learnrate'''
    #Class Constructor
    def __init__( self, epochs=10,learnrate=0.01):
        self.epochs = epochs
        self.learnrate = learnrate
    #Lets implement the following

    #Activation (sigmoid function)
    def _sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))

    #Ouptut(prediction function)
    def _prediction(self,features, weight,bias):
        '''Returns the probability of an event, which is always between 0 and 1.
           Parameters
           ----------
        features : the independent variables or predictors
        weight: Random initialised weight
        bias: Random initialised weight'''
        return self._sigmoid(np.dot(features,weight)+ bias)


    #error (log-loss function) for a single perceptron
    def _error_formula(self,y,output):
        '''Returns the value of the error, Which just takes the uncertainty of the predictions based on 
           how much it varies from the actual label.

           Parameters
           ----------
        y : The dependent variables or the target/label
        output: The prediction, which is between 0 and 1
        '''
        return -y*np.log(output) - (1-y)*np.log(1-output)

    #gradient descent step
    def _update_weights(self,x,y, weight,bias, learnrate):
        '''Returns the value of the updated weight and bias. 

           Parameters
           ----------
        x : The independent variables
        y : The dependent variables or the target/label
        weight: Random initialised weight
        bias: Random initialised weight
        learnrate : proportional steps to take for every iteration
        '''
        pred = self._prediction(x,weight,bias)
        error = y - pred
        weight += learnrate * error * x
        bias += learnrate * error
        return weight,bias

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
        errors = []
        last_loss = None
    
        #assign a random weight and bias zero
        n_records, n_features = features.shape
        self._weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
        self._bias = 0
        for iter in range(self.epochs):
            for x,y in zip(features,targets):
                output = self._prediction(x,self._weights,self._bias)
                error = y - output
                self._weights, self._bias = self._update_weights(x,y,self._weights,self._bias,self.learnrate)
                
        #print out log loss for training dataset 
        out = self._prediction(features,self._weights,self._bias)
        loss = np.mean(self._error_formula(targets,out))
        errors.append(loss)
        print('Training loss', loss)
        
    def predict_probab(self,features):
        out = self._prediction(features,self._weights,self._bias)
        return out
                
    def predict(self,features,targets,threshold=0.5):
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
        self._predictions = np.array([1 if values >= threshold else 0 for values in self.predict_probab(features)])
    
    def accuracy(self,targets):
        accuracy = np.mean(self._predictions==targets)
        print("Accuracy: ", accuracy)
