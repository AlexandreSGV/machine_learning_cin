import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV


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