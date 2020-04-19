Designing the logistic regression from scratch

logistic regression is a technique that is used to explain the relationship between the input variables(independent) and the output variable(dependent), what differentiates it from the normal linear regression is that the dependent variables can take only a fixed set of values ( 0 and 1) these values correspond to the classes of a classfication problem.

In logistic regression our goal is to identify the relationship betweeen the independent variables and dependent and we do these by estimating the probabilities using a logistics function (it is a sigmoid curve that is used to build the function with various parameters)

Building the logistic model

focus:
we building a model which take in features x1, x2, x3,x4,...,xn. and returns a binary output denoted by Y.

statistics:
let p be the probability of Y being 1 (i.e p = Prob(Y=1))
the variables relationship can be denoted as

ln(p/(1-p)) = b0 + b1x1 + b2x2 + b3x3 + b4x4 + bnxn

where
p/(1-p) denotes the likelihood of the event taking place.

ln(p/(1-p)) is the log of the likelihood of the event taking place and is used to represent the probability that lies between 0 and 1.

while terms b0, b1, b2, b3, b4,...,bn are the parameters that we are trying to estimate during training.

note: our dear interests is in getting the value of probability p in the above equation.

Solution:
1>> remove the log term on the LHS of the equation by raising the RHS as a power of e (exponential)
p/(1-p) = e^b0 + b1x1 + b2x2 + b3x3 + b4x4 +...+ bnxn

2>> simplify by cross multiplying to obtain the value of p
p = e^b0 + b1x1 + b2x2 + b3x3 + b4x4 +...+ bnxn / (1 +e^b0 + b1x1 + b2x2 + b3x3 + b4x4 +...+ bnxn)

this equation above can also known as the equation of the sigmoid function talked about earlier and we shall be using the above derived equation to make our predictions...

Implementation
L2 loss function was implemented to calculate the error and the Gradient Descent Algorithm was used to estimate the paramaters.

we shall be looking at the relationship between the Age of some patients and their Diabetes status to test the tested created.

Source of dataset
Microsoft: DAT263x Introduction to Artificial Intelligence (AI) Lab files
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VlglKoqggJKRM6EJP_opFRF3bW41iTBP' -O data.csv

