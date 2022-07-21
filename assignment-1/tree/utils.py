import pandas as pd
import numpy as np

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    elements,counts = np.unique(Y,return_counts = True)
    length = len(elements)
    entrpy = 0
    total = np.sum(counts)
    for i in range(length):
        entrpy += (-counts[i]/total)*np.log2(counts[i]/total)
    return entrpy

def variance(ar):
    ar=list(ar)
    if len(ar)==0:
        return 0
    return np.var(ar)

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    elements,counts = np.unique(Y,return_counts = True)
    l = len(elements)
    gini = 0.0
    total = np.sum(counts)
    for j in range(l):
        gini += (counts[j]/total)**2
    return 1-gini

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    attr = np.array(attr)
    assert(Y.size==attr.size)
    l = np.array(Y).size
    values, counts = np.unique(attr, return_counts=True)
    wt_entropy = 0
    num = Y.values[0]
    if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
        total_entropy = entropy(Y)
        for val in range(len(values)):
            wt_entropy += (counts[val]/l)*entropy(Y[attr==values[val]])
    else:
        total_entropy = variance(Y)
        for val in range(len(values)):
            wt_entropy += (counts[val]/l)*variance(Y[attr==values[val]])
    gain = total_entropy - wt_entropy
    return gain

def gain_with_gini(Y,attr):
    ''' Calculates information gain using gini index'''

    Y = np.array(Y)
    attr = np.array(attr)
    assert(Y.size==attr.size)
    l = np.array(Y).size
    values, counts = np.unique(attr, return_counts=True)
    gini = 0
    for j in range(len(values)):
        gini += (counts[j]/l)*gini_index(Y[attr==values[j]])
    gain = gini_index(Y) - gini
    return gain