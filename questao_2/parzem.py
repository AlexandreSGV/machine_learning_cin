
import pandas as pd
import numpy as np
from parzem_lib import *
from sklearn.model_selection import RepeatedStratifiedKFold


# df = pd.read_csv('image_segmentation_18_2098.csv', sep=';')
df = pd.read_csv('image_segmentation.csv', sep=';')

wNames = df.CLASSE.unique()
numeroClasses = len(wNames)

dados = df.iloc[:, 1:].values
gabarito = df['CLASSE'].values
print(gabarito)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=123456789)

# h = estimador_bandwidth(dados)
# print('bandwidth: ', h)
h = 6.98947320727

for indices_treinamento, indices_teste in rskf.split(dados, gabarito):
    conjunto_treinamento = dados[indices_treinamento]
    conjunto_teste = dados[indices_teste]
    gabarito_teste = gabarito[indices_teste]

    # print(conjunto_treinamento)
    # print(conjunto_teste.shape)
    # print(gabarito_teste)
    
    priori = calculatePrior(conjunto_treinamento, numeroClasses)
    c = 0
    # para cada elemento do treinamento, calcula a densidade
    for indice, elemento_teste in enumerate(conjunto_teste):
        classe_correta = gabarito[indice]

        predicted_class = predict(
            conjunto_treinamento,
            numeroClasses,
            elemento_teste, 
            h, 
            priori
        )
        print(predicted_class, classe_correta)

    # wCount[name] = len(df_treinamento[df_treinamento.CLASSE == name].values)
    # prioridadesPriori[name] = wCount[name] / len(df_treinamento.values)

# print(bandwidth_estimator(dados))