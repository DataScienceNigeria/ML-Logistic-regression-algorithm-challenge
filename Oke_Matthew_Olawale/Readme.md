This is a logistic regression algorithm built from scratch with the use of the numpy library only.

Logistic regression most times only have the capability for binary classification but this particular can also solve for multiclass classification using the OVR(One VS Rest Method.)

THE LOGISTIC REGRESSION ALGORITHM
- Initialize weights with its column as the number of features present in dataset

- Find the dot product of the weight and the data added with a particular bias which gives us our linear model.
                    (input_data * weights) + bias. 

- The sigmoid computation is then applied to this result to form probabilities(0 to 1) between predicted classes. 
                sigmoid_function  = 1/1(exp(-x)
        
- This helps find correlation between the input data and its weights from which error will be calculated and weigthts contributing to this error will be penalized by calculating the rate of change of error with weights and biases. 
            dw = (1 / number of input_data)*(input_data * error)
            db = (1 / number of input_data)*error
            where, error = predictions - true_label
    
- From here, weights are updated little by little by multiplying this rate of change with a particular fraction known as the learning rate and subtracting it from previous weights and biases

- This goes for a couple of iterations defaultly set "n_iters = 1000". This procees is called the gradient descent algorithm.

Multiclass Classification
- This is done by making binary classification on one of the classes against other classes repeatedly for each class.

- Sigmoid probabilties of their linear-model are computed.

- Then each sigmoid probabilities of these binary classifiers are extracted for each datapoint and passed through a softmax activation function. This also helps to put all values directly in probabilties between 0s and 1s making them sum up to 1.
            softmaxc_function = exp(xi)/ sum(exp(xi) , where xi = probability for each class for a particular input_data prediction
            
- Then the class with the final highest probability is chosen as the prediction.


For implementation:
model = LogisticRegression()   #This is implements for binary classification and automatically sets the multi parameter to False.
model.fit(X,y)
model.predict(X)
model.accuracy(y, y_predict)

------------------------------------------------------------------------------------------------------------------
model = LogisticRegression(multi=True)         #This is used for multiclass classification and "multi" is needed to be set "True".
