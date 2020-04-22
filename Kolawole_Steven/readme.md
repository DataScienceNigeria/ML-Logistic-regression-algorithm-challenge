A Customized Logistic Regression with Nesterov Accelerated Gradient with Early Stopping option:


Nesterov Accelerated Gradient is theorized to converge by at least, a 10 times faster rate than Stochastic Gradient Descent, and over 25 times faster rate than the naive batch gradient descent.

Nesterov Gradient combines the properties of Stochastic Gradient Descent -which supposedly have the properties that allows it to “jump” out of shallow local minima giving it a better chance of finding a true global minimum- with a 'smarter' momentum, that has a somewhat prescient notion of the global minimum, and knows to slow down before the hill slopes up again.

But there is a catch; 

Converging too fast makes it easier for the model to overfit, causing the well-known bias-variance tradeoff.

My way of avoiding that is to introduce Early Stopping, which works by simply terminating the iterations when there is no improvement in validation loss after a certain number of epochs.
