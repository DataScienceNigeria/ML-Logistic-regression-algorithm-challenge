# ML-Logistic-regression-algorithm-challenge
ML-Logistic-regression-algorithm-challenge
## Logistic Regression
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.The possible outcomes of a logistic regression are not numerical but
rather categorical ( 1 or 0, Yes or No ) etc.

Even though a Logistic regression is seen as a generalized linear model, It is a linear model with a link function that maps the output of linear multiple regression to a posterior probability of each class (0,1) using the **logistic
sigmoid function.**

## The Logistic Regression Formla


## $p(X)/1 −p (X)    = e ( β 0 +β 1 X 1 +...+β k X k )$

Where,

$p(X)$ = Probability of the dsitribution

$e$ = Base of the Natural Log

$β0$  = Biase or Intercept

$β1$  = Coefficient

$X1$ = Independenet variable

## **ODDS** = $p (X)/1 −p (X)$

The logistic regression model is not very useful in itself. The right-hand side of the model is an exponent which is very computationally inefficient and generally hard to grasp.

When we talk about a *logistic regression* what we usually mean is **logit regression** – which is a variation of the model where we have taken the log of both sides. See formula below:

## $log(p(X)/1 −p(X)) = log(e(β 0 + β 1 x + ⋯ β k x k))$

On the right hand side, log cancels 'e(exponential)' function leavig us with our new formula:

## $log(p(X)/1 −p(X)) = β 0 + β 1 x + ⋯ β k x k$

With odds:

## $log (odds) = β 0 + β 1 x + ⋯ β k x k$

We'll implemt all these in the code section of the project.

