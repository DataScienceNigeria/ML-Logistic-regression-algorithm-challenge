# Logistic Regression from scratch
The code is implemented in Python
## Methodology
Below are the methods used in writing Logistic Regression from scratch

1. Initialize a class Logistic Regression
Logistic Regression was used in the biological sciences in early twentieth century. It was then used in many social science applications. It is used when the dependent variable(target) is categorical.
For example,
**To predict whether an email is spam (1) or (0)
**Whether the tumor is malignant (1) or not (0)

It was initialized with attribute learning rate (lr), number of iteration(n_iters). 
The amount that the weights are updated during training is referred to as the step size or the “learning rate.”
The bias value allows the activation function to be shifted to the left or right, to better fit the data.

2. Sigmoid Function
The formula for linear function which produces continuous variables is;
        f(w,b) = wx + b
where w is the weight and b is bias and x is the data point. Since linear regression is used when the dependent variable(target) is categorical. Sigmoid function tends to convert the linear function to probalilities i.e 0 and 1.
Sigmoid function is 1 / (1 + e^-1)

3. Gradient Descent
Gradient Descent is used to iteratively update the weight, along with the learning rate to know how far the direction will go.

