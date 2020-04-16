
This logistic regression is to check the relationship between the sex of an individual and their height and weight

First we read in the CSV data and convert it into an array so that it can be easily worked upon

Standardizing the X values gives it a mean of 0 and a standard deviation of 1

One hot encoding simply converts the Y values to 0s and 1s, so that we can work with it

Then we split into test and train. We train over 70% of the data and then test it on the remaining data

We initialise the random weight we would need and set the bias to 0
 
We write the sigmoid function followed by the dot product needed for the sigmoid function.

The classification rate function gets the mean of Ytrain and predictions we calculated.
This gives up our classification score

The training cost and the test cost is calculated with the loss function (cross_entropy).
Then gradient descent on the weight and bias parameters.



 
