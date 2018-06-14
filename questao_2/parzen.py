import numpy as np
from common import *
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


def estimador_bandwidth(data):
    params = {'bandwidth': np.logspace(0.1, 1, 10)}
    grid = GridSearchCV(KernelDensity(), params, cv=10)
    grid.fit(data)
    return grid.best_estimator_.bandwidth

def regularization_function(h, x, x_i):
    return (x_i - x) / float(h)

# funcao de kernel multivariado produto
def gaussian_window_func(conjunto_treinamento, elemento_teste, h):
    x = regularization_function(h, elemento_teste, conjunto_treinamento)
    x = (1 / float(((2*np.pi)**0.5)*h)) * np.exp((-1/float(2))*(x)**2 )
    prod_sample= np.prod(x, axis=1)
    k = np.sum(prod_sample)
    return k

# kernel estimator
# conjunto_treinamento = training samples
# elemento_teste = test sample
# h = bandwidth
# d = number of dimensions
def estimador_parzen(conjunto_treinamento, elemento_teste, h):
    # print('shape conjunto_treinamento', conjunto_treinamento.shape)
    n = conjunto_treinamento.shape[0]
    v = h ** conjunto_treinamento.shape[1]
    k = gaussian_window_func(conjunto_treinamento, elemento_teste, h)

    density = (1/float(n)) * (1/float(v)) * float(k)

    return density

# naive bayes
#  calculate evidence from all sample likelihoods
def calcular_evidencia(likelihoods, priors):
    evidence = 0
    for classe, elemento in likelihoods.items():
        evidence = evidence + (elemento * priors[classe])
    return evidence

def calcula_posteriori(priori, densidade, evidencia):
    return np.float64(densidade * priori) / evidencia

def posterioris_por_classe(elemento_teste, conjunto_treinamento, h, priori):
    densidades = {}
    for classe, elementos in conjunto_treinamento.items():
        densidade = estimador_parzen(np.array(elementos), elemento_teste, h)
        densidades[classe] = densidade

    evidencia = calcular_evidencia(densidades, priori)

    # calculates posteriors
    posteriors = {}
    for classe, elementos in conjunto_treinamento.items():
        posterior = calcula_posteriori(priori[classe], densidades[classe],
                                       evidencia)
        posteriors[classe] = posterior

    return posteriors


def predictParzen(elemento_teste, conjunto_treinamento, h, priori):
    posteriors = posterioris_por_classe(elemento_teste, conjunto_treinamento, h, priori)

    return max(posteriors.keys(), key=(lambda k: posteriors[k]))


def executaParzen(dados, gabarito, nomeClasses, repeticoes, splits,
                  elementsByClass):



    rskf = RepeatedStratifiedKFold(
        n_splits=splits, n_repeats=repeticoes, random_state=123456789)
    print('shape dados', dados.shape)
    h = estimador_bandwidth(dados)
    print('bandwidth: ', h)
    # h = 6.98947320727
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



        priori = calculatePrior(conj_treinamento)

        errors = 0
        hits = 0
        estimativas = []
        # para cada elemento do treinamento, calcula a densidade
        for indice, elemento_teste in enumerate(conj_teste):
            classe_correta = gabarito_conj_teste[indice]

            predicted_class = predictParzen(elemento_teste, conj_treinamento, h, priori)

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
            if (gabarito_conj_teste[i] == estimativas[i]):
                acertos += 1
            else:
                erros += 1
        if (cont % splits == 0):
            acuracias.append(acertos / (acertos + erros))
            acertos = 0
            erros = 0


    return predictions, error_rates, acuracias
