Logistic Regression as the name implies is not a regression algorithm but a classification algorithm.Logistic Regression representation: 0<=sigmoid(Z)<=1Example of Logistic Regression use-cases are:
Classification of email as spam or not spam
Classification of breast cancer as Malignant or benignIn building logistic regression, i used the Wisconsin breast cancer dataset as the data to build the algorithm.LabelEncoder could have been used to convert my dependent variable 'Y' from string to integer but label encoder might make a mis-representation of the values i want for my classification and end up making a mistake with the prediction, so a simple 'if' statement worked the miracle since i wanedt my Malignant to be 1(i.e there is presence of breast cancer in the patient) and Benign to be 0(i.e there is absence of breast cancer in the patient)In Building a logistic regression we have to basically go through four processes:1. Building the sigmoid function: The sigmoid function is represented as g(Z)=1/1+e**-Z  but Z= transpose(theta).X : in order to

fit the parameter theta, we need to pick initial values for theta which is the reason for my function initializing theta as the parameters.
2. Building the cost function: usually looks like:
cost(sigmoid(Z), y)={-log(sigmoid(Z)) if y=1 and -log(i-sigmoid(Z)) if y=0 

but

it can be simplified to give us 

cost(sigmoid(Z), y) = -ylog(sigmoid(Z))-(1-y)log(1-sigmoid(Z))

for logistic regression, the cost function = 1/m* summation for all i=1 to m (cost(sigmoid(Z), y)) which will finally give us

-1/m[summation for all i=1 to m (ylog(sigmoid(Z))-(1-y)log(1-sigmoid(Z)))]3. Gradient Descent: In order to fit our parameters theta to the data, we need to minimize the cost function and in order to minimize the cost function, w emake use of the gradient descent.

it is represented as:

repeat{theta = theta - learning_rate * d/dtheta cost}

where d/dtheta cost = 1/m * summation for all i=1 to m(sigmoid(Z)-y)x

imputing it into thet theta formular, we have:
theta=theta-learning rate/m *summation for all i=1 to m(sigmoid(Z)-y)x

in order to implement it well, we make use of the vectorized implementation which looks like:
theta=theta-learning rate/m * transpose(X).(g(x.theta)-y)4. Finally, we build the prediction function

0.5 has been chosen as the threshhold

we just need to initialize our prediction and build a simple if conditional statement:
 
if sigmoid(Z)>=0.5:
	then prediction=1
else:
	prediction=0
In order to complete building the Logistic Regression algorithm, the 'logreg' function was defined, it made use of the already defined fuctions which was applied on the dataset to perform its function and xmake its prediction.NB: The cost function is used to penalize the algorithm for making a false prediction, in order to minimize this cost function,
we make use of gradient descent.We could decide to go further by:

5. Regularizing logistic regression cost
6. Regularizing logistic regression gradient

These can be achieved by using advanced optimization

but i didn't use it in the codes since they are just extra.Thanks to geekforgeeks and Stanford Machine Learning Course by Andrew Ng for the Knowledge of codes and mathematics imparted.