import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score



# calcula o valor de gama
def calcular_gama(dados):
    distancias = euclidean_distances(dados, dados, squared=True)  # calculando distância euclidiana
    lista_distancias = distancias[np.triu_indices(len(distancias), 1)] # transformando a matriz em uma lista
    lista_distancias = np.sort(lista_distancias)  # ordenando a lista das distâncias calculadas

    indice_quantil_1 = int((len(lista_distancias) * (1 / 10)) - 1)  # calculando o índice do quantil 0.1
    indice_quantil_9 = int((len(lista_distancias) * (9 / 10)) - 1)  # calculando o índice do quantil 0.9

    sigma = (lista_distancias[indice_quantil_1] + lista_distancias[indice_quantil_9]) / 2 # média do quantil 0.1 e 0.9
    gama = (1 / (sigma))
    return gama

# inicializa o vetor de hiperparametros com o valor de gama
def iniciar_hiperparametros(X, gama):
    hp = []
    for i in range(X.shape[1]):
        hp.append(gama)
    return hp

# inicializa o vetor de protótipos randomicamente
def iniciar_prototipos(c, dados):
    prototipos = []
    indices = []
    
    for i in range(c):
        indice = np.random.randint(0, len(dados)-1)
        prototipos.append(dados[indice])
        indices.append(indice)

        
    # print('indices: ',indices)
    return prototipos

def iniciar_clusters(c):
    clusters = []
    for i in range(c):
        clusters.append([])
    return clusters

# atribuição inicial dos objetos ao cluster
def iniciar_afetacao_objeto(c, dados, prototipos, hp):
    clusters = iniciar_clusters(c)
    for i in range(len(dados)):
        resultados = [] # cada objeto x tem uma lista com o resultado da variante KCM-K-GH considerando cada protótipo
        for p in range(len(prototipos)):
            resultados.append(2*(1 - calcular_variante_K(dados[i], prototipos[p], hp)))
        indice_menor_distancia = resultados.index(min(resultados))
        clusters[indice_menor_distancia].append(i)
    return clusters

### EQUAÇÕES ##

# calcula a variante KCM-K-GH para um objeto x, dado um protótipo p e um vetor de hiperparametros
def calcular_variante_K(x, p, hp): # Equação 9
    distancia = 0
    for j in range(len(hp)):
        distancia += hp[j] * np.power((x[j]-p[j]), 2)
    resultado = np.exp((-1.0/2.0)*distancia)
    return resultado

# calcula o representante do cluster (novos protótipos)
def calcular_representantes_clusters(dados, v_cluster, v_prototipos, v_hp): # Equação 14
    
    for c in range(len(v_prototipos)):
        if(len(v_cluster[c]) > 0):
            soma1, soma2 = 0, 0
            for e in range(len(v_cluster[c])):
                xk = dados[v_cluster[c][e]]
                gi = v_prototipos[c]
                soma1 += (calcular_variante_K(xk, gi, v_hp))*xk
                soma2 += (calcular_variante_K(xk, gi, v_hp))
            v_prototipos[c] = soma1/soma2
    return v_prototipos

# REVER ESSE CÓDIGO
# calcula o vetor de hiperparametros
def calcular_hiperparametros(X, prototipos, clusters, gama, hiper): # Equação 16
    hparametros = []
    produto = 1
    for h in range(X.shape[1]):
        soma1 = 0
        for c in range(len(clusters)):
            soma2 = 0
            for k in range(len(clusters[c])):
                xk = X[clusters[c][k]]
                gi = prototipos[c]
                parte1 = calcular_variante_K(xk, gi, hiper)
                parte2 = np.power(xk[h]-gi[h], 2)
                soma2 += parte1*parte2
            soma1 += soma2
        produto *= soma1
    resultado1 = np.power(produto, 1/X.shape[1])*gama
        
    
    for j in range(X.shape[1]):        
        # calculando resultado 2
        soma1 = 0
        for w in range(len(clusters)):
            soma2 = 0
            for k in range(len(clusters[w])):
                xk = X[clusters[w][k]]
                gi = prototipos[w]
                parte1 = calcular_variante_K(xk, gi, hiper)
                parte2 = np.power(xk[j] - gi[j], 2)
                soma2 += parte1*parte2
            soma1 += soma2
        resultado2 = soma1
        # print('res1 ', resultado1, ' res2 : ', resultado2)
        hparametros.append(resultado1/(resultado2))
    return hparametros

# para cada objeto x de X
# etapa de alocação - equação 18
def atribuir_objeto_cluster(x, v_prototipos, v_hp):
    resultado = []
    for p in range(len(v_prototipos)):
        # print('variante K', calcular_variante_K((x), v_prototipos[p], v_hp))
        # print('2*(1 - variante K)', 2.0*(1.0-calcular_variante_K((x), v_prototipos[p], v_hp)))
        resultado.append(2*(1-calcular_variante_K((x), v_prototipos[p], v_hp)))
    # print('Resultado : ', resultado)
    indice_distancia_menor = resultado.index(min(resultado))
    return indice_distancia_menor

# busca o cluster atual de um objeto
# recebe o indice do objeto x de X e uma lista c/ os clusters
def retorna_indice_atual(i, clusters):
    indice_atual = 0
    for c in range(len(clusters)):
        if ((i in clusters[c]) == True):
            indice_atual = c
            break
    return indice_atual

# para cada objeto x de X
# etapa de alocação - equação 18
def funcao_objetivo(X, v_clusters, v_prototipos, v_hp):
    resultado = 0
    for i in range(len(v_clusters)):
        for k in range(len(v_clusters[i])):
            resultado += 2*(1 - calcular_variante_K(X[v_clusters[i][k]] , v_prototipos[i],v_hp))
    return resultado

# Calcular o indice de Rand
def indice_rand(X,Y, v_clusters):
    labels_pred = []
    labels_true = []
    for i in range(len(X)):
        for c in range(len(v_clusters)):
            if (i in v_clusters[c]) == True:
                labels_pred.append(c)
                break
    for j in range(len(Y)):
        for h in range(len(Y[j])):
            labels_true.append(Y[j][h])

    return adjusted_rand_score(labels_true, labels_pred)