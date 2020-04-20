# ML-Logistic-regression-algorithm-challenge


![DSN logo](DSN_logo.png)|DSN Algorithm Challenge|
|---|---|

I used the Numpy Library to create the Logistic Regression from scratch
I also tested the algorithm and compared it with the Logistic Regression algorithm provided by scikit learn
The matplotlib library is for visualization.


# My Method
I created a class Logistic Regression and it has four (4) argument
argument 1 :  lr short for learning_rate which will determine how my algorithm will move not interms of direction but in weights learned during training a model
argument 2: num_iter short for Number of Iterations this is the number of times the algorithm will go over the data in order to learn the weights and if it is too high it takes time to finish training and it may lead to overfitting and too low will lead to underfitting
argument 3: fit_intercept this is based on the assumtion that the data has linear relationship among them (features and target) hence it uses the intercept learned to predict probability
argument 4: verbose which is to output the training steps

Logistic Regression is to model the probability of a certain class or event existing such as win or loss

The sigmoid function therefore allows this to be possible because the sigmoid function maps any real value into another value between 0 and 1.
