import pandas as pd
import numpy as np
import codigo_modulado as code
import time

dados = pd.read_csv('data/segmentation_18_col.csv', sep=',')
# X = dados.iloc[:, 1:].values # conjunto de dados
df = dados.iloc[:, 1:]
# df = df.drop(df.columns[10], axis=1)
# df = df.drop(df.columns[10], axis=1)
# df = df.drop(df.columns[10], axis=1)
print(df.columns)
X = df.values # conjunto de dados

# X = dados.iloc[:, 10:].values # visão RGB
Y = dados.iloc[:, 0:1].values
# print(y)
c = 7
melhor_valor_func_objetivo = 99999999
melhor_indice_rand = 0
melhor_hiperparametros = []
melhor_prototipos = []
melhor_clusters = []
melhor_execucoes = 0
melhor_quantidades = []
for execucoes in range(2):

    gama = code.calcular_gama(X)
    hiperparametros = code.iniciar_hiperparametros(X, gama)

    prototipos = code.iniciar_prototipos(c, X)

    clusters = code.iniciar_afetacao_objeto(c, X, prototipos, hiperparametros)
    rodadas = 0
    quantidades = []
    teste = 99999

    start_total_time = time.time()
    while (teste > 300):
        # step 1
        prototipos = code.calcular_representantes_clusters(X, clusters, prototipos, hiperparametros)
        # step 2
        hiperparametros = code.calcular_hiperparametros(X, prototipos, clusters, gama, hiperparametros)

        # step 3
        teste = 0
        for i in range(len(X)):
            indice_atual = code.retorna_indice_atual(i, clusters)
            indice_afetado = code.atribuir_objeto_cluster(X[i], prototipos, hiperparametros) # equação 18
            if (indice_atual != indice_afetado):
                teste += 1
                clusters[indice_atual].remove(i)
                clusters[indice_afetado].append(i)
        rodadas+=1



        valor_func_objetivo = code.funcao_objetivo(X, clusters, prototipos, hiperparametros)
        quantidades = []
        for i in range(len(clusters)):
            quantidades.append(len(clusters[i]))
        # print('Execução : ', execucoes,' | Rodadas : ', rodadas, '| teste : ', teste)
        print('Execução : ', execucoes,' | Rodadas : ', rodadas, '| teste : ', teste,  '| : ', quantidades,' | FuncObjetivo : ',valor_func_objetivo,' | ',(time.time() - start_total_time), 'seconds')
        indice_rand = code.indice_rand(X,Y, clusters)
        print('INDICE DE RAND', indice_rand)

    if(valor_func_objetivo < melhor_valor_func_objetivo):
        melhor_valor_func_objetivo = valor_func_objetivo
        melhor_indice_rand = indice_rand
        melhor_hiperparametros = hiperparametros
        melhor_prototipos = prototipos
        melhor_clusters = clusters
        melhor_execucoes = execucoes
        melhor_quantidades = quantidades



print('######################')
print('MELHOR EXECUÇÃO : ', melhor_execucoes)
print('PARTIÇÃO')
for i in range(len(melhor_clusters)):
    print('\tOBJETOS GRUPO ', i)
    print(melhor_clusters[i])
print('RERPESENTANTES DE CADA GRUPO')
for i in range(len(melhor_prototipos)):
    indice = np.nonzero(X == melhor_prototipos[i])
    print('\tProtótipo ', indice, ' : ', melhor_prototipos[i])
print('NUMERO DE OBJETOS DE CADA GRUPO')
for i in range(len(melhor_quantidades)):
    print('\tGrupo ', i,' : ', melhor_quantidades[i], ' elementos.')
print('VETOR DE HYPERPARAMETROS')
for i in range(len(melhor_hiperparametros)):
    print('\tHyperparâmetro ', i,' : ', melhor_hiperparametros[i])
print('ÍNDICE DE RAND CORRIGIDO')
print('\tÍndice de Rand corrigido: ',melhor_indice_rand)
print('FUNÇÃO OBJETIVO')
print('\tFunação Objetivo: ',melhor_valor_func_objetivo)
print('######################')
