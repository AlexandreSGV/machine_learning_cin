import numpy as np
import bigfloat
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold

# X grupo em que se quer calcular a média
def calcula_mi(X):
    X = np.array(X)
    # print('X', X)
    soma = np.zeros(X.shape[1])


    for i in range(len(X)):
        soma += X[i]
    soma /= len(X)

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

def estimar_parametros(conj_treinamento):
    mis = {}
    sigmas = {}
    for classe, elementos in conj_treinamento.items():
        mis[classe] = calcula_mi(elementos)
        sigmas[classe] = calcula_sigma(elementos, mis[classe])
    return mis, sigmas

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

def posterioris_por_classe(elemento_teste, prioris, mis, sigmas):
    probabilidades = {}
    for classe in prioris:
        posteriori = calcula_posteriori(elemento_teste, mis[classe],
                                        sigmas[classe])
        parte1 = posteriori * prioris[classe]
        parte2 = 0
        for r in mis.keys():
            parte2 += calcula_posteriori(elemento_teste, mis[r],
                                         sigmas[r]) * prioris[r]
        probabilidades[classe] = parte1 / parte2

    return probabilidades


def predictGauss(elemento_teste, prioris, mis, sigmas):

    probabilidades = posterioris_por_classe(elemento_teste, prioris, mis, sigmas)

    return max(probabilidades.keys(), key=(lambda k: probabilidades[k]))

def executaGaussiana(dados, gabarito, nomeClasses, repeticoes, splits):

    rskf = RepeatedStratifiedKFold(
        n_splits=splits, n_repeats=repeticoes, random_state=123456789)


    cont = 0
    acertos = 0
    erros = 0
    acuracias = []

    predictions = []
    error_rates = []

    for indices_treinamento, indices_teste in rskf.split(dados, gabarito):
        cont += 1
        print('cont ', cont)
        conj_treinamento = {}
        conj_teste = dados[indices_teste]
        gabarito_conj_teste = gabarito[indices_teste]

        for name in nomeClasses:
            conj_treinamento[name] = []

        for i in indices_treinamento:
            conj_treinamento[gabarito[i]].append(dados[i])


        probabilidadesPriori = calculatePrior(conj_treinamento)
        mis, sigmas = estimar_parametros(conj_treinamento)


        estimativas = []
        errors = 0
        hits = 0
        repetition_predictions = []
        for indice, elemento_teste in enumerate(conj_teste):


            predicted_class = predictGauss( elemento_teste, probabilidadesPriori, mis, sigmas)
            classe_correta = gabarito_conj_teste[indice]


            # usado para gerar matriz de confusao
            prediction = []
            # prediction.append(nomeClasses.tolist().index(classe_correta))
            # prediction.append(nomeClasses.tolist().index(predicted_class))
            prediction.append(classe_correta)
            prediction.append(predicted_class)
            predictions.append(prediction)
            # repetition_predictions.append(prediction)
            if (predicted_class == classe_correta):
                hits += 1
            else:
                errors += 1

            # usado para gerar tabela de acusária de 1 a N Repetions
            estimativas.append(predicted_class)

        error_rate = errors / (hits + errors)
        error_rates.append(error_rate)
        # predictions.append(repetition_predictions)

        for i in range(len(gabarito_conj_teste)):
            # print(gabarito_conj_teste[i], ' - ', estimativas[i], ' - ', gabarito_conj_teste[i] == estimativas[i])
            if (gabarito_conj_teste[i] == estimativas[i]):
                acertos += 1
            else:
                erros += 1
        if (cont % splits == 0):
            acuracias.append(acertos / (acertos + erros))
            acertos = 0
            erros = 0
    return predictions, error_rates, acuracias