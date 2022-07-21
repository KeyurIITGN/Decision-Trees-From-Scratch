"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
#from linearRegression.linearRegression import LinearRegression

np.random.seed(42)



N = 50
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
tree = DecisionTreeClassifier(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(train_x, train_y)
y_hat = Classifier_AB.predict(test_x)
[fig1, fig2] = Classifier_AB.plot()

print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, test_y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, test_y, cls))
    print('Recall: ', recall(y_hat, test_y, cls))

#comparing it with a decision stump
tree.fit(train_x,train_y)
y_hat2=tree.predict(test_x)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat2, test_y))
for cls in y.unique():
    print('Precision: ', precision(y_hat2, test_y, cls))
    print('Recall: ', recall(y_hat2, test_y, cls))
