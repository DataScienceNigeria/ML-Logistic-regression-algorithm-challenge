# olawale_abdulrasheed_olamide
consider a dataset with multiple feature these features are related, this relation wed refer to as weight , 
the relation between these features lets us determine its outcome(in this case ,classifying).
logistic regression is a widely used algorithm for classification.
i'll be walking through the steps that makes it acomplish it's goals.
Making use of sigmoid fuction a logistic regression predicts an outcome between 0 and 1.
but in the case of our classification , it is to be used to predict wheather or not some features fit in a particular group (0 or 1). taking a threshold of 0.5 (personal preference) and groping based on that , any feature less than the threshold is to be grouped seperately from one greater than or equal to the threshold.
our fuction determined, we can now determine our weights (relationship of the features,some are more important than others ,hence larger weight)
now,the problem is getting the best weights since we have no idea what it is.  we cant brute force our way through (thats alot), but we can use gradient descent which its name suggest we descend gradually till minimal and almost optimal weights ,using its loss function ,gradually descendimg with its learning rate(so as not to overstep the minimum).by comparing the loss and continuosly adjusting the weights to minimize this loss (number of time you check and update is to be determined by you ofcourse)
