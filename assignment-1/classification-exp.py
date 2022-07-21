import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
# print(X)
X=pd.DataFrame(X)
#print(X.head())
y=pd.Series(y)
for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X.iloc[:70,:], y.iloc[:70:])
    y_hat = tree.predict(X.iloc[71:,:])
    print("accuracy: ",accuracy(y_hat,y.iloc[71::]))
    print('Criteria :', criteria)
    for i in y.unique():            # gives per class precision and recall
        print("precision: ",precision(y_hat,y.iloc[71::],i))
        print("recall: ",recall(y_hat,y.iloc[71::],i))

X['y']=y
# 5-fold crossvalidation for the generated data
def fivefoldcv(X):
    l=X.shape[0]
    fold_acc=[]
    depths=[]
    #outer loop to divide test and train
    for i in range(5):
        fold_train=X.iloc[:round(l*(0.8)),:]
        fold_test=X.iloc[round(l*(0.8)):,:]
        ar=np.array_split(fold_train,5)
        acc_arr=[]
        #train and validation split
        for j in range(5):
            val_data=ar[0]
            val_y=val_data['y']
            val_X=val_data.drop(['y'],axis=1)
            frames=[]
            for k in range(j):
                if k!=j:
                    frames.append(ar[k])
            train_data=pd.concat(ar)
            train_y = train_data['y']
            train_X = train_data.drop(['y'], axis=1)
            # print(train_X)
            # print(train_y)
            # train the model with different depths and test it on the validation set
            depth_acc = [0]		
            for m in range(1, 11):
                tree = DecisionTree('inforamation_gain', m)
                tree.fit(train_X, train_y)
                y_hat = tree.predict(val_X)
                acc = accuracy(y_hat, val_y)
                depth_acc.append(acc)
            acc_arr.append(depth_acc)
        # calulate average accuracy for each depth
        l = len(acc_arr)
        avg_accs = [0]
        for t in range(1,11):
            temp = []
            for j in range(5):
                temp.append(acc_arr[j][t])
            avg = np.mean(temp)
            avg_accs.append(avg)
        for k in range(1,11):
            if(avg_accs[k]>avg_accs[k-1]):
                print("depth: ", k, ' --> accuracy: ', avg_accs[k])
        
        # take the best depth from validation
        best_depth = 10
        max_acc = 0
        for k in range(1,11):
            if(avg_accs[k]>max_acc):
                max_acc = avg_accs[k]
                best_depth = k
        
        #finding the fold accuracy
        testing_y=fold_train['y']
        testing_X=fold_train.drop(['y'],axis=1)
        final_y=fold_test['y']
        final_x=fold_test.drop(['y'],axis=1)
        tree = DecisionTree('inforamation_gain', best_depth)
        tree.fit(testing_X, testing_y)
        y_final = tree.predict(final_x)
        ac = accuracy(y_final, final_y)
        fold_acc.append(ac)
        depths.append(best_depth)
	
    # print(fold_accuracies)
    return fold_acc, depths
print(fivefoldcv(X))
