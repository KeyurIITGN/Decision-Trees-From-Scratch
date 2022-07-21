
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

df=pd.read_csv("C:\\Users\\Keyur\\github-classroom\\ES654\\assignment-1-KeyurIITGN\\assignment-1\\auto-mpg.csv")
df['horsepower']=pd.to_numeric(df.horsepower,errors='coerce')
df=df.dropna()
df.pop("car name")
X=df.drop('mpg', axis=1)
y=df['mpg']
train_x=X.iloc[:int(0.7*df.shape[0]),:]
test_x=X.iloc[int(0.7*X.shape[0])+1:,:]
train_y=y.iloc[:int(0.7*y.shape[0])]
test_y=y.iloc[int(0.7*X.shape[0])+1:]
tree = DecisionTree(criterion='gini_index') #Split based on gini index
tree.fit(train_x, train_y)
y_hat = tree.predict(test_x)
# print(test_y)
print("My implementation: ")
print('RMSE: ', rmse(y_hat, test_y))
print('MAE: ', mae(y_hat, test_y))
tree_sklearn=DecisionTreeRegressor()
tree_sklearn.fit(train_x,train_y)
y_hat_sklearn=tree_sklearn.predict(test_x)
print("Sklearn implementation: ")
print('RMSE: ', rmse(y_hat, test_y))
print('MAE: ', mae(y_hat, test_y))