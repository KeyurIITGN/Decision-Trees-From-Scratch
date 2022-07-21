import pandas as pd
import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(len(y_hat) == len(y))
    y_hat=list(y_hat)
    y=list(y)
    length=len(y)
    similar=0
    for i in range(length):
        if y_hat[i]==y[i]:
            similar+=1
    return similar/length

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(len(y_hat) == len(y))
    y_hat=list(y_hat)
    y=list(y)
    match=0
    pred_match=0
    for i in range(len(y)):
        if y_hat[i]==cls:
            if y[i]==cls:
                match+=1
            pred_match+=1
    if pred_match==0:
        return 1
    return match/pred_match

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(len(y_hat) ==y.size)
    y_hat=list(y_hat)
    y=list(y)
    match=0
    data_match=0
    for i in range(len(y)):
        if y[i]==cls:
            if y_hat[i]==cls:
                match+=1
            data_match+=1
    if data_match==0:
        return 1
    return match/data_match

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(len(y_hat)==len(y))
    sum1=0.0
    y_hat=list(y_hat)
    y=list(y)
    length=len(y)
    for i in range(length):
        sum1+=(y_hat[i]-y[i])**2
    mse=sum1/length
    return np.sqrt(mse)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(len(y_hat)==len(y))
    sum1=0.0
    y_hat=list(y_hat)
    y=list(y)
    length=len(y)
    for i in range(length):
        sum1+=np.abs(y_hat[i]-y[i])
    mae=sum1/length
    return mae
