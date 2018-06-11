import numpy as np
import pandas as pd
from common_lib import *
import nayve_bayes as bayes
import parzen
import gaussian
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.decomposition import PCA

repeticoes = 2
splits = 2


def predicaoCombinada(priors, posteriors):
    sub = 1 - 3
    multip = sub * priors
    sum_rule = multip + posteriors
    sum_rule = np.sum(sum_rule, axis=1)
    return np.argmax(sum_rule)


# number of classes
numberOfClasses = 10
# patterns per class
patternSpace = 200

df = pd.read_csv('data/segmentation_18_col.csv', sep=',')
df = df.sort_values('CLASSE')
nomeClasses = df.CLASSE.unique()

# Get datasets as a numpy 2d array
completeViewData = df.iloc[:, 1:].values
shapeViewData = df.iloc[:, 1:9].values
rgbViewData = df.iloc[:, 9:19].values

completeViewData = preprocessing.scale(completeViewData)
shapeViewData = preprocessing.scale(shapeViewData)
rgbViewData = preprocessing.scale(rgbViewData)

# project the d-dimensional data to a lower dimension
pca = PCA(n_components=15, whiten=False)
completeViewData = pca.fit_transform(completeViewData)
shapeViewData = pca.fit_transform(shapeViewData)
rgbViewData = pca.fit_transform(rgbViewData)

# Generates numpy array of targets (classes)
gabarito = df.CLASSE.values

# stratified cross validation
rskf = RepeatedStratifiedKFold(
    n_splits=splits, n_repeats=repeticoes, random_state=123456789)
#skf = StratifiedKFold(n_splits=10, random_state=42)
'''FIXED BANDWIDTHS'''
completeView_h = 1.9952
shapeView_h = 2.3865
rgbView_h = 1.9952

print("completeView best bandwidth: {0}".format(completeView_h))
print("shapeViewData best bandwidth: {0}".format(shapeView_h))
print("rgbView best bandwidth: {0}".format(rgbView_h))

r = 0
error_rates = []
predictions = []
for indices_treinamento, indices_teste in rskf.split(completeViewData, gabarito):

    r += 1
    gabarito_conj_teste = gabarito[indices_teste]


    conj_treinamentoCompleteView = {}
    conj_treinamentoShapeView = {}
    conj_treinamentoRGBView = {}
    conj_testeCompleteView = completeViewData[indices_teste]

    for name in nomeClasses:
        conj_treinamentoCompleteView[name] = []
        conj_treinamentoShapeView[name] = []
        conj_treinamentoRGBView[name] = []

    for i in indices_treinamento:
        conj_treinamentoCompleteView[gabarito[i]].append(completeViewData[i])
        conj_treinamentoShapeView[gabarito[i]].append(shapeViewData[i])
        conj_treinamentoRGBView[gabarito[i]].append(rgbViewData[i])

    #print("repetition %s" % r)

    # training set and class sample size - same for all datasets
    # train_sample_size = conj_treinamentoCompleteView.shape[0]
    # test_sample_size = completeView_test_set.shape[0]
    # train_class_size = train_sample_size / numberOfClasses
    '''UNCOMMENT FOR NEW BANDWIDTH ESTIMATIONS'''
    # completeView_h = parzen.bandwidth_estimator(conj_treinamentoCompleteView)
    #print("completeView best bandwidth: {0}".format(completeView_h))
    # shapeView_h = parzen.bandwidth_estimator(conj_treinamentoShapeView)
    #print("shapeViewData best bandwidth: {0}".format(shapeView_h))
    # rgbView_h = parzen.bandwidth_estimator(conj_treinamentoRGBView)
    #print("rgbView best bandwidth: {0}".format(rgbView_h))

    # compute priors - same for all datasets
    prior_ = bayes.calculatePrior(conj_treinamentoCompleteView, numberOfClasses)

    # computes theta for all classes
    completeView_mu_, completeView_sigma_ = gaussian.estimateParameters(conj_treinamentoCompleteView,
                                                      numberOfClasses)
    shapeView_mu_, shapeView_sigma_ = gaussian.estimateParameters(conj_treinamentoShapeView,
                                                      numberOfClasses)
    rgbView_mu_, rgbView_sigma_ = gaussian.estimateParameters(conj_treinamentoRGBView,
                                                      numberOfClasses)

    # predict class for each sample in test set
    true_positives = 0
    false_positives = 0
    samples_predictions = []
    for sample_index in range(0, test_sample_size):

        completeView_test_sample = completeView_test_set[sample_index]
        shapeView_test_sample = shapeView_test_set[sample_index]
        rgbView_test_sample = rgbView_test_set[sample_index]

        # true class for sample - is the same true class for all views
        actual_class = completeView_test_target[sample_index]

        # afeta exemplo a classe de maior posteriori
        # gaussian classifier
        # calculate posteriors from view completeView using gaussian classifier
        completeView_gauss_posteriors = gaussian.posteriorFromEachClass(
            completeView_test_sample, completeView_mu_, completeView_sigma_, prior_)

        # calculate posteriors from view shapeViewData using gaussian classifier
        shapeView_gauss_posteriors = gaussian.posteriorFromEachClass(
            shapeView_test_sample, shapeView_mu_, shapeView_sigma_, prior_)

        # calculate posteriors from view rgbView using gaussian classifier
        rgbView_gauss_posteriors = gaussian.posteriorFromEachClass(
            rgbView_test_sample, rgbView_mu_, rgbView_sigma_, prior_)

        # calculate posteriors from view completeView using parzen window
        completeView_parzen_posteriors = parzen.posteriorFromEachClass(
            conj_treinamentoCompleteView, train_class_size, completeView_test_sample, completeView_h, prior_)

        # calculate posteriors from view shapeViewData using parzen window
        shapeView_parzen_posteriors = parzen.posteriorFromEachClass(
            conj_treinamentoShapeView, train_class_size, shapeView_test_sample, shapeView_h, prior_)

        # calculate posteriors from vew rgbView using parzen window
        rgbView_parzen_posteriors = parzen.posteriorFromEachClass(
            conj_treinamentoRGBView, train_class_size, rgbView_test_sample, rgbView_h, prior_)

        posteriors = zip(completeView_gauss_posteriors, shapeView_gauss_posteriors,
                         rgbView_gauss_posteriors, completeView_parzen_posteriors,
                         shapeView_parzen_posteriors, rgbView_parzen_posteriors)

        # Ensemble classifiers using the sum rule
        predicted_class = predicaoCombinada(prior_, posteriors)

        #print("actual:", actual_class, " prediction:", predicted_class)
        #print("true class: %s" % actual_class)
        #print("predicted: %s"% predicted_class)

        # usado para gerar matriz de confusao
        prediction = []
        prediction.append(actual_class)
        prediction.append(predicted_class)
        samples_predictions.append(prediction)

        # para calculo de taxa de erro
        if (actual_class == predicted_class):
            true_positives += 1
        else:
            false_positives += 1

    error_rate = util.errorRate(true_positives, false_positives)

    error_rates.append(error_rate)

    predictions.append(samples_predictions)

print("confusion matrix")
confusionMatrix = util.confusionMatrix(predictions)
print(np.array_str(confusionMatrix, precision=6, suppress_small=True))
print("")

print("precision by class")
precision_by_class = confusionMatrix.diagonal() / float(
    patternSpace * repeticoes)
print(precision_by_class)
print("")

print("precision average %s" % np.mean(precision_by_class))
print("error rate average %s" % util.errorRateAverage(error_rates))
print("")

# erro rate for each repetition
for i, rate in enumerate(error_rates):
    print("repetition:", i, "error rate", rate)
print("")
