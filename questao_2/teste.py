import pandas as pd
import numpy as np
from scipy.linalg import expm, sinm, cosm
# from sklearn.metrics.cluster import
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

nb =  GaussianNB()

from sklearn.model_selection import RepeatedStratifiedKFold
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
              [13, 14, 15], [16, 17, 18] ])
y = np.array([1, 1, 1, 0, 0, 0])
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=4, random_state=36851234)
rep = 0
for train_index, test_index in rskf.split(X, y):
    if (rep % 3 == 0):
        print('Repetição ', int(rep / 3))
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rep +=1
print('############################')
for i in range(3):
    print('Repetição ' , i)
    kfold = StratifiedKFold(n_splits=3, random_state=36851234)
    # print(kfold)
    for train_index, test_index in kfold.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
