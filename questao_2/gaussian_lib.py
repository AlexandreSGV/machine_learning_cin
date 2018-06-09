import numpy as np
import bigfloat
from sklearn.model_selection import RepeatedStratifiedKFold

# X grupo em que se quer calcular a média
def calcula_mi(X):
    X = np.array(X)
    # print('X', X)
    soma = np.zeros(X.shape[1])


    for i in range(len(X)):
        # print('X[i]', X[i])
        soma += X[i]
    soma /= len(X)
    # print('mi', soma)

    return soma

# X é vetor com elementos do grupo (shape NxD)
# mi é média dos elementos do grupo
def calcula_sigma(X, mi):
    X = np.array(X)
    soma = np.zeros((X.shape[1],X.shape[1]))
    n = len(X)

    for i in range(n):
        xk = X[i].reshape(len(X[i]), 1)
        xkt = X[i].reshape(1, xk.shape[0])
        soma += np.dot(xk, xkt)        

    mii = mi.reshape(len(mi), 1)
    miit = mi.reshape(1, mii.shape[0])

    soma -= n * np.dot(mii, miit)
    soma /= n

    soma = soma * np.identity(soma.shape[1])
    return soma



def calcula_posteriori(xk, mi, sigma):
    d = mi.shape[0]
    pi = ((2 * np.pi)**(-d/2))
    determinante =  (np.linalg.det( np.linalg.pinv(sigma)) ** 0.5)
    xk = xk.reshape(1,len(xk))
    mi = mi.reshape(1,len(mi))

    diff = (xk - mi)
    parte3 = diff.reshape(1,diff.shape[1])

    parte1 = diff.reshape(diff.shape[1],1)

    parte2 = np.linalg.pinv(sigma)

    mult1 = np.dot(parte2,parte1)

    mult2 = np.dot(parte3, mult1)

    exponencial = np.exp( (-1/2) * mult2 )

    
    return pi * determinante * exponencial

# falta testar
def calcula_bayesianoGaussiano(xk, wName,prioridadesPriori,mis,sigmas):
    # print('wName',wName)
    # print('mis[wName]',mis[wName])
    # print('sigmas[wName]',sigmas[wName])
    # print('prioridadesPriori[wName]',prioridadesPriori[wName])
    posteriori = calcula_posteriori(xk,mis[wName],sigmas[wName])
    # print('posteriori ',posteriori)
    parte1 = posteriori * prioridadesPriori[wName]
    # print('parte1 - ', parte1)
    parte2 = 0
    for name in mis.keys():
        parte2 += calcula_posteriori(xk,mis[name],sigmas[name]) *  prioridadesPriori[name]

    return parte1 / parte2

def executaGaussiana(dados, classes, nomeClasses, repeticoes, splits):

    rskf = RepeatedStratifiedKFold(
        n_splits=splits, n_repeats=repeticoes, random_state=123456789)


    cont = 0
    acertos = 0
    erros = 0
    acuracias = []
    for indices_treinamento, indices_teste in rskf.split(dados, classes):
        cont += 1
        print('cont ', cont)
        conj_treinamento = {}
        conj_teste = dados[indices_teste]
        gabarito_conj_teste = classes[indices_teste]

        for name in nomeClasses:
            conj_treinamento[name] = []

        for i in indices_treinamento:
            conj_treinamento[classes[i]].append(dados[i])

        qtdClasses = {}
        probabilidadesPriori = {}
        mis = {}
        sigmas = {}
        for classe, elementos in conj_treinamento.items():
            # print('classe',classe)
            qtdClasses[classe] = len(elementos)
            probabilidadesPriori[
                classe] = qtdClasses[classe] / len(indices_treinamento)
            mis[classe] = calcula_mi(elementos)
            sigmas[classe] = calcula_sigma(elementos, mis[classe])

        estimativas = []
        for xk in conj_teste:
            probabilidades = {}
            for name in nomeClasses:
                probabilidades[name] = calcula_bayesianoGaussiano(
                    xk, name, probabilidadesPriori, mis, sigmas)

            estimativas.append(
                max(probabilidades.keys(), key=(lambda k: probabilidades[k])))

        for i in range(len(gabarito_conj_teste)):
            # print(gabarito_conj_teste[i], ' - ', estimativas[i], ' - ', gabarito_conj_teste[i] == estimativas[i])
            if (gabarito_conj_teste[i] == estimativas[i]):
                acertos += 1
            else:
                erros += 1
        if (cont % splits == 0):
            # print('Repetição ', int(cont / splits))
            # print('Tamanho conjunto de testes : ', acertos+erros)
            # print('acertos : ', acertos)
            # print('erros : ', erros)
            # print('acurácia : ', acertos / (acertos + erros))
            acuracias.append(acertos / (acertos + erros))
            acertos = 0
            erros = 0
    return acuracias