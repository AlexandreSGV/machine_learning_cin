import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


def estimador_bandwidth(data):
    params = {'bandwidth': np.logspace(0.3, 1, 10)}
    grid = GridSearchCV(KernelDensity(), params, cv=10)
    grid.fit(data)
    return grid.best_estimator_.bandwidth

# calculate prior for classes
def calculatePrior(train_set, numberOfClasses):
    prior_ = []
    # print(train_set)
    # iterates through class training samples
    for w in range(0, numberOfClasses):

        # training set and class sample size
        train_sample_size = train_set.shape[0]
        class_sample_size = train_sample_size / numberOfClasses

        #compute prior
        prior = class_sample_size / float(train_sample_size)
        prior_.append(prior)

    return prior_

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
    n = conjunto_treinamento.shape[0]
    v = h ** conjunto_treinamento.shape[1]
    k = gaussian_window_func(conjunto_treinamento, elemento_teste, h)

    density = (1/float(n)) * (1/float(v)) * float(k)

    #print("k", k)
    #print("n", n)
    #print("v", float(v)
    #print("density", density)

    return density

# naive bayes
#  calculate evidence from all sample likelihoods
def calcular_evidencia(likelihoods, priors):
    evidence = 0
    for i in range(0, likelihoods.shape[0]):
        evidence = evidence + (likelihoods[i] * priors[i])
    return evidence

def calcula_posteriori(priori, densidade, evidencia):
    return (densidade * priori) / evidencia

def predict(conjunto_treinamento, numeroClasses, elemento_teste, h, priori):
    size = int(conjunto_treinamento.shape[0] / numeroClasses)

    densidades = []
    for w in range(0, numeroClasses):
        # limits for training samples from class
        train_initial_sample = w * size
        # print('start ',train_initial_sample)
        train_end_sample = train_initial_sample + size
        # print('end', train_end_sample)
        train_samples = conjunto_treinamento[train_initial_sample:train_end_sample, :]
        # print('samples', train_samples)

        # calculates density through parzen window estimation with gaussian kernel
        densidade = estimador_parzen(train_samples, elemento_teste, h)
        densidades.append(densidade)

    # print('densidades', densidades)

    densidades = np.array(densidades)

    # calculate the evidence

    evidencia = calcular_evidencia(densidades, priori)
    # print('evidencia', evidencia)

    # calculates posteriors
    # w given x
    posteriors = []
    for w in range(0, numeroClasses):
        posterior = calcula_posteriori(priori[w], densidades[w], evidencia)
        posteriors.append(posterior)
        # print("prior", prior_[w], "density", densities[w], "evidence", evidence)
        # print("w", w, "posterior", posterior)

    # print("posteriors", posteriors)
    # print('maximum', np.argmax(posteriors))
    return np.argmax(posteriors)

def generateTargets(numberOfClasses, patternSpace):
    target_train = []
    for i in range(0, numberOfClasses):
        target_train.append([i] * patternSpace)

    return np.hstack(target_train)


def calculateConfusionMatrix(repetitions_predictions):
    matrix = np.zeros((7, 7))
    for predictions in repetitions_predictions:
        for prediction in predictions:
            matrix[prediction[0], prediction[1]] += 1
    return matrix


def executaParzen(dados, gabarito, nomeClasses, repeticoes, splits,
                  elementsByClass):


    numeroClasses = len(nomeClasses)
    target = generateTargets(len(nomeClasses), elementsByClass)


    rskf = RepeatedStratifiedKFold(
        n_splits=splits, n_repeats=repeticoes, random_state=123456789)

    # h = estimador_bandwidth(dados)
    # print('bandwidth: ', h)
    h = 6.98947320727
    cont = 0
    acertos = 0
    erros = 0
    acuracias = []

    predictions = []
    error_rates = []
    # print('target', target)
    for indices_treinamento, indices_teste in rskf.split(dados, target):
        cont += 1
        print('cont ', cont)
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
        # c = 0
        errors = 0
        hits = 0
        repetition_predictions = []
        estimativas = []
        # para cada elemento do treinamento, calcula a densidade
        for indice, elemento_teste in enumerate(conjunto_teste):
            classe_correta = gabarito_teste[indice]

            predicted_class = predict(conjunto_treinamento, numeroClasses,
                                    elemento_teste, h, priori)

            # usado para gerar matriz de confusao
            prediction = []
            prediction.append(classe_correta)
            prediction.append(predicted_class)
            repetition_predictions.append(prediction)
            if (predicted_class == classe_correta):
                hits += 1
            else:
                errors += 1

            # usado para gerar tabela de acusária de 1 a N Repetions
            estimativas.append(predicted_class)


        error_rate = errors / (hits + errors)

        error_rates.append(error_rate)
        predictions.append(repetition_predictions)
        for i in range(len(gabarito_teste)):
            # print(gabarito_teste[i], ' - ', estimativas[i], ' - ',
                #   gabarito_teste[i] == estimativas[i])
            if (gabarito_teste[i] == estimativas[i]):
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


    return predictions, error_rates, acuracias
