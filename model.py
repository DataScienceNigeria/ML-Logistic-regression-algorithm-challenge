import numpy as np

class logistic_regression:
    
    def __init__(self,iterations,alpha):
        self.iterations=iterations  #choosing the number of iterations (Hyperparameter)
        self.alpha=alpha       #choosing alpha(Hyperparameter) 
    
    def sigmoid(self,z):
        return(1/(1+np.exp(-z)))    #sigmoid function
    
    def fit(self,x,y):              #(X-data for training, y - Output) 
        m=x.shape[0]                
        self.w=np.random.randn(x.shape[1],1)  #Initializing the weight
        
        cost_vals=[] 
        for i in range(2):     #For each number of iterations
            a= np.dot(x,self.w)            #multiplying the weights with the Feature values and summing them up
            z=self.sigmoid(a)         #Using link function to transform the data
            
            cost = (-1/m) *( np.dot(y,np.log(z))+(np.dot((1-y),np.log(1-z))))  #Calculating the cost function
            
            cost_vals.append(cost)        #Creating a list with all cost values for each iteration
            
            dw = np.dot(x.T,z-np.array([y])).mean()  #Calculating the gradient
            
            self.w=self.w-(self.alpha*dw)         #updating the weights
        return self
    
    def predict(self,x,threshold): 
        probability=self.sigmoid(np.dot(x,self.w))  #predicting a new set of values based on the training 

        if(probability>threshold):
            return (1)
        else:
            return (0)


if __name__ == "__main__":
    logistic_regression()