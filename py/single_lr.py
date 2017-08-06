#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Try to solve the real problem using single variable linear regression model
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


matplotlib.use("Agg")
boston = load_boston()
# print(boston.DESCR)

X, y = boston.data, boston.target

print("Take a look at the features information...")
print("features names: {}".format(boston.feature_names))
print(X[:5, ])
print("And response variable:")
print(y[:5])

# Then lets checkout the variable relation with y
data = pd.DataFrame(X, columns=boston.feature_names)

# Take a look at whole data alike.
print(data.describe())

# Take room variable with house price
Xs = X[:, 5]


def plotData(X, y):
    
    plt.scatter(X, y, s=80, c="red", marker="+", alpha=.5)
    plt.xlabel("Room Number")
    plt.ylabel("Boston House Price")
    
plotData(Xs, y)


