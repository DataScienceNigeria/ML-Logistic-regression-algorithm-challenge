import numpy as np
from logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report,confusion_matrix

#ad_data = pd.read_csv('advertising.csv')
#predictors from advertising data
#X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
#y = ad_data['Clicked on Ad']

bc_dataset = datasets.load_breast_cancer()
X, y = bc_dataset.data, bc_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


log = LogisticRegression(lr = 0.1, n_iterations=1000)
log.fit(X_train,y_train)
predictions = log.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print(accuracy(y_test, predictions))
print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))