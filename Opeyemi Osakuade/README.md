## Design and develop the Logistic Regression algorithm from scratch using Python
Logistic regression is a classification algorithm,used to model the probability of a certain class or event. It transforms the output into a probability value (i.e. a number between 0 and 1) using logistic sigmoid function.

For a binary classifier, we want the classifier to output values that are between 0 and 1. 
<br /> 
i.e.
                                  0 &le; y<sub>&theta;</sub>(x)&le;1

### Hypothesis Representation

The term logistic regression refers to "logit function" which refers to "log odds". Odds refers to the ratio of the probability of an event occuring to the probability it does not occur.
Taking the log, log odds for the model turns out to be the equation of the *Sigmoid Function*

### Cost function
Since the logistic regression function(sigmoid) is *non linear*, to get a *convex function*, i.e a bowl-shaped function that eases the gradient descent function's work to converge to the optimal minimum point,a logistic regression cost function is derived

### Gradient descent
To choose the values of weights that corresponds to a convex function and fits the data well(so we reach a global minimum), ensure that 

the prediction(h) is at least close to the actual *y*, minimize the cost function using gradient descent.

Repeat until convergence, updating all weights.

#### Data
Comprise of two written test scores at DMV driving school and also contains the result whether passed or failed, the objective is to predict if each person with the test scores passed or failed
