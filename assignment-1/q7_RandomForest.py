"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

########### RandomForestClassifier ###################

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

X['y'] = y
X = X.sample(frac=1.0)
y = X['y']
X = X.drop(['y'], axis=1)

train_x=X.iloc[:int(0.6*X.shape[0]),:]
test_x=X.iloc[int(0.6*X.shape[0])+1:,:]
train_y=y.iloc[:int(0.6*y.shape[0])]
test_y=y.iloc[int(0.6*X.shape[0])+1:]


for criteria in ['information_gain', 'gini_index']:
    Classifier_RF = RandomForestClassifier(10, criterion = criteria)
    Classifier_RF.fit(train_x, train_y)
    #print("Hello")
    y_hat = Classifier_RF.predict(test_x)
    Classifier_RF.plot()
    # print("hello")
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, test_y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, test_y, cls))
        print('Recall: ', recall(y_hat, test_y, cls))

########### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

X['y'] = y
X = X.sample(frac=1.0)
y = X['y']
X = X.drop(['y'], axis=1)

train_x=X.iloc[:int(0.6*X.shape[0]),:]
test_x=X.iloc[int(0.6*X.shape[0])+1:,:]
train_y=y.iloc[:int(0.6*y.shape[0])]
test_y=y.iloc[int(0.6*X.shape[0])+1:]

Regressor_RF = RandomForestRegressor(10, criterion = criteria)
Regressor_RF.fit(train_x, train_y)
y_hat = Regressor_RF.predict(test_x)
Regressor_RF.plot()
print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, test_y))
print('MAE: ', mae(y_hat, test_y))
