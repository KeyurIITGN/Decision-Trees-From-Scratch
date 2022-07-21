import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

###Write code here

N = 30
P = 2
NUM_OP_CLASSES = 2
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

train_x=X.iloc[:int(0.6*X.shape[0]),:]
test_x=X.iloc[int(0.6*X.shape[0])+1:,:]
train_y=y.iloc[:int(0.6*y.shape[0])]
test_y=y.iloc[int(0.6*X.shape[0])+1:]
n_estimators = 3
Classifier = RandomForestClassifier(n_estimators=n_estimators,criterion='information_gain')
Classifier.fit(train_x, train_y)
y_hat = Classifier.predict(test_x)
Classifier.plot()
# [fig1, fig2] = Classifier_AB.plot()
print('Accuracy: ', accuracy(y_hat, test_y))
for cls in set(test_y):
    print('Precision: ', precision(y_hat, test_y, cls))
    print('Recall: ', recall(y_hat, test_y, cls))