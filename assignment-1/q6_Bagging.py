"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
# Or use sklearn decision tree
#from linearRegression.linearRegression import LinearRegression

########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

train_x=X.iloc[:int(0.6*X.shape[0]),:]
test_x=X.iloc[int(0.6*X.shape[0])+1:,:]
train_y=y.iloc[:int(0.6*y.shape[0])]
test_y=y.iloc[int(0.6*X.shape[0])+1:]

criteria = 'information_gain'
tree = DecisionTreeClassifier(criterion=criteria)
Classifier_B = BaggingClassifier(base_estimator=DecisionTreeClassifier, n_estimators=n_estimators )
Classifier_B.fit(train_x, train_y)
y_hat = Classifier_B.predict(test_x)
[fig1, fig2] = Classifier_B.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, test_y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, test_y, cls))
    print('Recall: ', recall(y_hat, test_y, cls))
