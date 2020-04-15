#importing needed modules
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def preparing_data():
    df=pd.read_csv('weight-height.csv')#reading in the data
    df_matrix=df.values   #converting it into an array

    
    Y=df_matrix[:,0]   # Y(output) becomes the gender feature
    X=df_matrix[:,1:].astype(np.float) # X becomes the remaining features 
    
    #standardizing X variables
    for i in (0, 1):
        m = X[:, i].mean()
        s = X[:, i].std()
        X[:, i] = (X[:, i] - m) / s

    #one_hot encoding for Y
    # males=0 , females=1
    N=len(Y)
    Y2=np.array([0]*N)

    for n in range(N):
        g = Y[n]
        Y2[n]

        if g == 'Male':
            Y2[n] = 0
        else:
            Y2[n] = 1

    Y=Y2

    return X, Y


X, Y=preparing_data()
X,Y=shuffle(X,Y)

#test and train
Xtrain=X[:7000] 
Ytrain=Y[:7000]

Xtest=X[-2000:]
Ytest=Y[-2000:]



N,D=Xtrain.shape
w= np.random.randn(D) #weight
b=0 #bias



#logistic function
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

#dot product for the logistic function
def forward(X,w,b):
    return sigmoid(X.dot(w)+b)

def classification_rate(Y, P):
    return np.mean(Y==P)

P_Y_given_X = forward(Xtrain,w,b)

predictions= np.round(P_Y_given_X)
print(" Classification Score is: ", classification_rate(Ytrain, predictions))


#loss function
def cross_entropy(T, P):
    return -np.mean(T*np.log(P)+(1-T)*np.log(1-P))

train_cost=[]

test_cost=[]
learning_rate = 0.001

for i in range(10000):
    pYtrain=forward(Xtrain, w, b)
    pYtest=forward(Xtest,w, b)

    ctrain= cross_entropy(Ytrain, pYtrain)
    ctest=cross_entropy(Ytest,pYtest)

    train_cost.append(ctrain)
    test_cost.append(ctest)

    w-= learning_rate*Xtrain.T.dot(pYtrain-Ytrain)
    b-=learning_rate*(pYtrain-Ytrain).sum()
    

print('final train classification rate: ', classification_rate(Ytrain, np.round(pYtrain)))
print('final test classification rate: ', classification_rate(Ytest, np.round(pYtest)))



print('Final Weight: ', w)
legend_train,=plt.plot(train_cost, label="train cost")
legent_test,= plt.plot(test_cost, label="test cost")
plt.legend([legend_train, legent_test]);
plt.show()
