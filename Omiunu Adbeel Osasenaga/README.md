# Developing Logistic Regression Algorithm From Scratch

Here i show Logistic Regression can be developed from scratch. Firstly Logistic regression is a generalized linear model that we can use to model or predict cases where the target variable is categorical by type. So simply put Logistic Regression Models are Classification models. They are used to predict the category or class an element or entity falls into. For the case where we have to classify a set of variables into two classes we call that Binary Classification Problem. For binary classification case the model(in this case Logistic Regression) predict the target variable by producing outputs of 0 when it belongs to a class and 1 when it belongs to the other class.

## Approximations
Logistic Regression function is derived from the transformation of a Linear Regression function using what is known as a sigmoid function.            
           
### 1. Generating the Logistic Regression from Linear Regression
First the linear regression is given below as:

![image1 showing Linear Regression](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/1.Linear%20Regression%20Equation.png)

where **w represents the weights, x represents the matrix of features or the input vector and b represents the bias**

Recall that inorder to transform the above function we make use of the sigmoid function. The sigmoid function is shown below: 

![image2 showing sigmoid function](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/2.%20sigmoid%20function.png)

Its graphically represented as:

![image3 showing the graphical representation of the sigmoid function](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/3.%20Sigmoid%20plot.png)

The purpose of the sigmoid function is to transform the above linear regression into generating output values as in form probabilities of that fall between 0 and 1. 
Even after this tranformation the outputs do not yet reflect what the typical classification output should look like. So, we set a threshold is set at x = 0.5, such that all values greater than and equal to 0.5 are approximated as 1 and others less than this threshold are rounded down as 0.

Substituting the linear regression equation into the sigmoid function, satisfying the above conditions for x= 0.5 as the threshold. We have new sigmoid function to be:

![image4 showing the transformed linear regression](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/4.%20Substituted%20Sigmoid%20function.png)

on substituting, x from the sigmoid function equals (wx + b) from the linear regression fumction
where     **h(x)** is the transformed linear regression function, which is the logistic regression function 
          **y_hat** is the predicted value of the target varible


### 2. Cost Function
The cost function we use here is the Cross Entropy Cost Function. It is shown mathematically as:

![image5 showing the Cross Entropy Cost Function](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/5.%20Cost%20Function.png)

#### What is a COST FUNCTION?
The cost function measures how poorly a model does in terms of its ability to estimate the relationship between the independent variable(X) and the dependent variable (y). The goal therefore in this case is minimise or reduce the cost function. In reducing the Cost function what we are doin in essence is reducing the weights or finding what value of weights and biases would minimise the cost function and in turn optimise our Machine Learning model. 
To minimise the cost function by applying something called a **Gradient descent** on the cost function. The gradient descent is simply a partial differentation applied of the cost function with respect to the weights and biases.

![image6 showing the Gradient Descent on the weight](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/6.%20Gradient%20Descent.png)

![image7 showing the difference between a big learning rate versus small learning rate](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/7.%20Learning%20Rate.png)

On applying gradient descent the updated weights, biases and cost function becomes:

![image8 showing updated weights, biases and cost functions](https://raw.githubusercontent.com/Adbe-El/Develop-Logistic-Regression-From-Scratch/master/Images/8.%20Updated%20Weights.png)

'''**Note: J'= updated cost functions
        dJ/dw = dw
        dJ/db = db
        N = number of samples**'''

## Code Description 

1. A class **LogisticRegression** was created.
          '''class LogisticRegression:'''

2. An iniialisation function is created to intitialise all parameters namely: **learning rate(lr), number of iterations (n_iters), biases and weights.**
                      '''def __init__(self, learning_rate=0.001, n_iters=1000):
                                 self.lr = learning_rate
                                 self.n_iters = n_iters
                                 self.weights = None
                                 self.bias = None'''

3. A function called **_sigmoid** is created that returns the mathematical convention of the sigmoid function as cited above.
          
            '''def _sigmoid(self, x):
                    return 1 / (1 + np.exp(-x))'''

4. We define a function called **predict** that takes a parameter X(the independent variables). Within the function
- We assign the mathematical estimation for the linear regression model to the variable name **linear_model**
- apply the **sigmoid** function to the **linear_model** and assigned it the variable name **y_predicted**
- we define the threshold within our sigmoid function for returning values of 0 and ! assigned to the variable name **y_predicted_cls** and return the array of the resulting value. 
          
          '''def predict(self, X):
                      linear_model = np.dot(X, self.weights) + self.bias
                      y_predicted = self._sigmoid(linear_model) 
        
                      y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
                      return np.array(y_predicted_cls)'''

5. A function called **fit** is created. Within this function  
- we assign the number of samples(n_samples) and the number of features(n_features) values equivalent to the number rows and columns of our independent variable(X) respectively.
- the weights are initialised using an array of zeroes following the dimension of n_features and bias is initialised with zero.
           
          '''def fit(self, X, y):
               n_samples, n_features = X.shape
               self.weights = np.zeros(n_features)
               self.bias = 0'''

- a **for loop** is created following the iterative process of the gradient descent of the cost function. The number of times the loop is set to run given by the value  **n_iters**.
- Within the **for loop** we have the following:
- We assign the mathematical estimation for the linear regression model to the variable name **linear_model**
- apply the **sigmoid** function to the **linear_model** and assigned it the variable name **y_predicted**
                      
           '''for _ in range(self.n_iters):
                      linear_model = np.dot(X, self.weights) + self.bias
                      y_predicted = self._sigmoid(linear_model)'''

                      
                      - compute the gradients of both weights and bias, given as **dw** and **db** given by the equation.
                      - update the weights and bias using.
                      
                 '''dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db'''

6. Finally, we test our model on the breast cancer data set from sklearn package.
