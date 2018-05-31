
import pandas as pd
import numpy as np
from parzem_lib import *
from sklearn.model_selection import RepeatedStratifiedKFold


# df = pd.read_csv('image_segmentation_18_2098.csv', sep=';')
df = pd.read_csv('image_segmentation_ordenados.csv', sep=';')

wNames = df.CLASSE.unique()
numeroClasses = len(wNames)

dados = df.values
dados = df.iloc[:, 1:].values
gabarito = df['CLASSE'].values
# print(gabarito)
target = generateTargets(len(wNames), 300)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=123456789)

# h = estimador_bandwidth(dados)
# print('bandwidth: ', h)
h = 6.98947320727

for indices_treinamento, indices_teste in rskf.split(dados, target):
    
    conjunto_treinamento = dados[indices_treinamento]
    # print(conjunto_treinamento[ : , 1: ]) # sem a coluna da classe
    # print(conjunto_treinamento[ : , :1 ]) #somente a coluna da classe

    # conjunto_teste = dados[indices_teste][ : , 1: ]
    # gabarito_teste = dados[indices_teste][ : , :1 ]
    conjunto_teste = dados[indices_teste]
    gabarito_teste = target[indices_teste]
    
    

    # print(conjunto_treinamento)
    # print(conjunto_teste.shape)
    # print(gabarito_teste.shape)
    
    priori = calculatePrior(conjunto_treinamento, numeroClasses)
    c = 0
    erros = 0
    acertos = 0
    # para cada elemento do treinamento, calcula a densidade
    for indice, elemento_teste in enumerate(conjunto_teste):
        classe_correta = gabarito_teste[indice]

        predicted_class = predict(
            conjunto_treinamento,
            numeroClasses,
            elemento_teste, 
            h, 
            priori
        )
        
        # print(wNames[predicted_class], classe_correta , wNames[predicted_class] == classe_correta)
        if (predicted_class == classe_correta):
            acertos+=1
        else:
            erros+=1
    print('acertos', acertos)
    print('erros', erros)
    print('acuracia',acertos/(erros+acertos))


    # wCount[name] = len(df_treinamento[df_treinamento.CLASSE == name].values)
    # prioridadesPriori[name] = wCount[name] / len(df_treinamento.values)

# print(bandwidth_estimator(dados))