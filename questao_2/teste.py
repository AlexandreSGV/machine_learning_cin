import pandas as pd
import numpy as np
from scipy.linalg import expm, sinm, cosm
# from sklearn.metrics.cluster import
from sklearn.naive_bayes import GaussianNB

nb =  GaussianNB()

from sklearn.model_selection import RepeatedStratifiedKFold
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
              [13, 14, 15], [16, 17, 18] ])
y = np.array([1, 1, 1, 0, 0, 0])
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=36851234)
for train_index, test_index in rskf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
