import operator
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
df_teste = dados.drop(df_treinamento.index)






w = {}
wCount = {}
prioridadesPriori  = {}
mis = {}
sigmas  = {}
# Inicializa variáveis


# for i in range(7):
#     print(np.linalg.det(np.linalg.pinv(np.identity(18) * gnb.sigma_[i])))
for name in wNames:    
    w[name] = df_treinamento[df_treinamento.CLASSE == name].iloc[:, 1:].values
    
    wCount[name] = len(df_treinamento[df_treinamento.CLASSE == name].values)
    prioridadesPriori[name] = wCount[name] / len(df_treinamento.values)
    # calcula paramentros
    mis[name] = calcula_mi(w[name])
    sigmas[name] = calcula_sigma(w[name],mis[name])



gabarito = df_teste['CLASSE'].values
# print(gabarito)
estimativa = []
for xk in df_teste.iloc[:, 1:].values:
    # print("#################")
    probabilidades = {}
    for name in w.keys():
        p = calcula_bayesianoGaussiano(xk, name,prioridadesPriori,mis,sigmas)
        probabilidades[name] = p
        # print('p[',name,'] ' , "%.15f" %p[0][0])
    # key, value = max(probabilidades.iteritems(), key=lambda x:x[1])
    # print('Max : ',  max(probabilidades.keys(), key=(lambda k: probabilidades[k])))
    estimativa.append(max(probabilidades.keys(), key=(lambda k: probabilidades[k])))

acertos = 0
erros = 0
for i in range(len(gabarito)):
    print (gabarito[i] , ' - ', estimativa[i] , ' - ', gabarito[i] == estimativa[i]) 
    if(gabarito[i] == estimativa[i]):
        acertos +=1
    else:
        erros +=1

print('Tamanho conjunto de testes : ', len(df_teste.values))
print('acertos : ', acertos)
print('erros : ', erros)
print('acurácia : ', acertos/len(gabarito))


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


# Código compara sigmas da GaussianNB e dos feitos pela fórmula do projeto
# teste = df_treinamento.iloc[:, 1:].values
# classes = df_treinamento['CLASSE'].values
# gnb.fit(teste, classes)

# for i in range(7):
#     print(np.linalg.det(np.linalg.pinv( np.identity(18) *gnb.sigma_[i])))
# print('#####################')
# for name in wNames:
#     print (np.linalg.det(np.linalg.pinv( np.identity(18) * sigmas[name].diagonal())))