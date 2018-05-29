import pandas as pd
from questao2_lib import *
import numpy as np
import codigo_modulado as code
import time
from sklearn.naive_bayes import GaussianNB

gnb =  GaussianNB()

dados = pd.read_csv('image_segmentation_18_2098.csv', sep=';')
wNames = dados.CLASSE.unique()
print (wNames)

# Percentual usado para treinamento
porcentagemTreinamento = 80

# seleciona elementos do grupo de treinamento aleatoriamente
df_treinamento = dados.sample(int(round( (len(dados.values) * porcentagemTreinamento)/100)))
# grupo de teste são os elementos que não estao em treinamento
df_teste = dados.drop(df_treinamento.index).iloc[:, 1:].values





teste = []
w = {}
wCount = {}
prioridadesPriori  = {}
mis = {}
sigmas  = {}
# Inicializa variáveis
for name in wNames:    
    w[name] = df_treinamento[df_treinamento.CLASSE == name].iloc[:, 1:].values
    wCount[name] = len(df_treinamento[df_treinamento.CLASSE == name].values)
    prioridadesPriori[name] = wCount[name] / len(df_treinamento.values)
    # calcula paramentros
    mis[name] = calcula_mi(w[name])
    sigmas[name] = calcula_sigma(w[name],mis[name])




for xk in df_teste:
    print("#################")
    for name in w.keys():
        p = calcula_bayesianoGaussiano(xk, name,prioridadesPriori,mis,sigmas)
        print('p[',name,'] ' , p)





# X = dados.iloc[:, 1:].values # conjunto de dados
# df = dados.iloc[:, 1:]
# print(df.columns)
# X = df.values # conjunto de dados

# Y = dados.iloc[:, 0:1].values
# print(y)
# c = 7





# mu = calcula_mi(X)
# sigma = calcula_sigma(X, mu)
# for i in range(1):
#     print('############################')
#     print(calcula_posteriori(X[i], mu, sigma))

# print(mu.shape)
# print(len(mu))
# print(mu)
# print(sigma, sigma.shape)
# print(mu.transpose())

# a = np.array([2,4,6,8])
# b = np.array([1,1,1,1,1,2])
# c = calcula_transpose(a)

