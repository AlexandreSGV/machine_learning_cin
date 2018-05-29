import pandas as pd
import numpy as np
from scipy.linalg import expm, sinm, cosm
# from sklearn.metrics.cluster import 
from sklearn.naive_bayes import GaussianNB

nb =  GaussianNB()

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array(['aaa', 'bbb', 'ccc', 'dddd', 'eeee', 'fff'])
nb.fit(X, Y)
print(nb)

# var = (np.power(-4, 1/2))
# # var = np.power(-0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000298638265773 , 0.5) 
# print (var)

# # print(-1.48206539355**0.5)
# a = np.array([1,2,3,4,5,6])
# # print(a.reshape(1,6) )

a = [
            [1,2,4],
            [4,5,6],
            [7,8,9]]

b = [
            [1,2,4],
            [4,5,6],
            [7,8,9]]

# print('exp ' , np.exp(1/2*np.dot(a,b)))


# print('Det a ',np.linalg.det(a))
# b = np.linalg.inv(a)
# c = np.linalg.pinv(a)
# print('Det b ',np.linalg.det(b))
# print(b)
# print('Det c ',np.linalg.det(c))
# print(c)