import numpy as np
import pandas as pd
import common as common
import parzen
import gaussian
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
import time
from pandas_ml import ConfusionMatrix
startTotalTime = time.time()

repeticoes = 30
splits = 10


def predicaoCombinada(prioris, completeView_gauss_posteriors,
                      shapeView_gauss_posteriors, rgbView_gauss_posteriors,
                      completeView_parzen_posteriors,
                      shapeView_parzen_posteriors, rgbView_parzen_posteriors):
    sub = 1 - 6
    soma = {}

    for classe in prioris:
        soma[classe] = prioris[classe]*sub + completeView_gauss_posteriors[classe] + shapeView_gauss_posteriors[classe] + rgbView_gauss_posteriors[classe] + completeView_parzen_posteriors[classe] + shapeView_parzen_posteriors[classe] + rgbView_parzen_posteriors[classe]

    # print('soma', soma)
    # somatorio = 0
    # for classe in soma:
    # somatorio += soma[classe]
    # print('somatorio', somatorio)
    return max(soma.keys(), key=(lambda k: soma[k]))





# patterns per class
elementsByClass = 300

df = pd.read_csv('data/segmentation_18_col.csv', sep=',')
df = df.sort_values('CLASSE')
nomeClasses = df.CLASSE.unique()

# Get datasets as a numpy 2d array
completeViewData = df.iloc[:, 1:].values
shapeViewData = df.iloc[:, 1:9].values
rgbViewData = df.iloc[:, 9:19].values

# completeViewData = preprocessing.scale(completeViewData)
# shapeViewData = preprocessing.scale(shapeViewData)
# rgbViewData = preprocessing.scale(rgbViewData)

# project the d-dimensional data to a lower dimension
# pca = PCA(n_components=15, whiten=False)
# completeViewData = pca.fit_transform(completeViewData)
# shapeViewData = pca.fit_transform(shapeViewData)
# rgbViewData = pca.fit_transform(rgbViewData)

# Generates numpy array of targets (classes)
gabarito = df.CLASSE.values

# variável armazenara os resultados em cada viev (col: view x row: repetition)
resultados = {}

# stratified cross validation
rskf = RepeatedStratifiedKFold(
    n_splits=splits, n_repeats=repeticoes, random_state=123456789)
#skf = StratifiedKFold(n_splits=10, random_state=42)
'''FIXED BANDWIDTHS'''
completeView_h = 10
shapeView_h = 10
rgbView_h = 1.9952

print("completeView best bandwidth: {0}".format(completeView_h))
print("shapeViewData best bandwidth: {0}".format(shapeView_h))
print("rgbView best bandwidth: {0}".format(rgbView_h))

cont = 0
acertos = 0
erros = 0
acuracias = []

error_rates = []
predictions = []
for indices_treinamento, indices_teste in rskf.split(completeViewData, gabarito):
    cont+=1
    print('repetição', cont)

    gabarito_conj_teste = gabarito[indices_teste]


    conj_treinamentoCompleteView = {}
    conj_treinamentoShapeView = {}
    conj_treinamentoRGBView = {}
    conj_testeCompleteView = completeViewData[indices_teste]
    conj_testeShapeView = shapeViewData[indices_teste]
    conj_testeRgbView = rgbViewData[indices_teste]

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

    # completeView_h = parzen.estimador_bandwidth(
    #     completeViewData[indices_treinamento])
    # print("completeView best bandwidth: {0}".format(completeView_h))
    # shapeView_h = parzen.estimador_bandwidth(
    #     shapeViewData[indices_treinamento])
    # print("shapeViewData best bandwidth: {0}".format(shapeView_h))
    # rgbView_h = parzen.estimador_bandwidth(
    #     rgbViewData[indices_treinamento])
    # print("rgbView best bandwidth: {0}".format(rgbView_h))

    # compute priors - same for all datasets
    prioris = common.calculatePrior(conj_treinamentoCompleteView)

    # computes theta for all classes
    completeView_mis, completeView_sigmas = gaussian.estimar_parametros(conj_treinamentoCompleteView)
    shapeView_mis, shapeView_sigmas = gaussian.estimar_parametros( conj_treinamentoShapeView)
    rgbView_mis, rgbView_sigmas = gaussian.estimar_parametros( conj_treinamentoRGBView)

    estimativas = []
    # predict class for each sample in test set
    true_positives = 0
    false_positives = 0
    samples_predictions = []

    # for i_elemento_teste in range(0, test_sample_size):
    for i_elemento_teste in range(0,len(conj_testeCompleteView)):

        completeView_elemento_teste = conj_testeCompleteView[i_elemento_teste]
        shapeView_elemento_teste = conj_testeShapeView[i_elemento_teste]
        rgbView_elemento_teste = conj_testeRgbView[i_elemento_teste]

        # true class for sample - is the same true class for all views
        classe_correta = gabarito_conj_teste[i_elemento_teste]

        # afeta exemplo a classe de maior posteriori
        # gaussian classifier
        # calculate posteriors from view completeView using gaussian classifier
        completeView_gauss_posteriors = gaussian.posterioris_por_classe( completeView_elemento_teste, prioris, completeView_mis, completeView_sigmas)

        # calculate posteriors from view shapeViewData using gaussian classifier
        shapeView_gauss_posteriors = gaussian.posterioris_por_classe( shapeView_elemento_teste, prioris, shapeView_mis, shapeView_sigmas)

        # calculate posteriors from view rgbView using gaussian classifier
        rgbView_gauss_posteriors = gaussian.posterioris_por_classe( rgbView_elemento_teste, prioris, rgbView_mis, rgbView_sigmas)




        # calculate posteriors from view completeView using parzen window
        completeView_parzen_posteriors = parzen.posterioris_por_classe( completeView_elemento_teste, conj_treinamentoCompleteView, completeView_h, prioris)


        # calculate posteriors from view shapeViewData using parzen window
        shapeView_parzen_posteriors = parzen.posterioris_por_classe( shapeView_elemento_teste, conj_treinamentoShapeView, shapeView_h, prioris)


        # calculate posteriors from vew rgbView using parzen window
        rgbView_parzen_posteriors = parzen.posterioris_por_classe( rgbView_elemento_teste, conj_treinamentoRGBView, rgbView_h, prioris)

        # print(completeView_gauss_posteriors)

        # posteriors = zip(completeView_gauss_posteriors, shapeView_gauss_posteriors,
        #                  rgbView_gauss_posteriors, completeView_parzen_posteriors,
        #                  shapeView_parzen_posteriors, rgbView_parzen_posteriors)

        # posteriors = set(
        #     completeView_gauss_posteriors & shapeView_gauss_posteriors &
        #     rgbView_gauss_posteriors & completeView_parzen_posteriors &
        #     shapeView_parzen_posteriors & rgbView_parzen_posteriors)

        # posteriors = common_entries(
        #     completeView_gauss_posteriors, shapeView_gauss_posteriors,
        #     rgbView_gauss_posteriors, completeView_parzen_posteriors,
        #     shapeView_parzen_posteriors, rgbView_parzen_posteriors)

        predicted_class = predicaoCombinada(
            prioris, completeView_gauss_posteriors, shapeView_gauss_posteriors,
            rgbView_gauss_posteriors, completeView_parzen_posteriors,
            shapeView_parzen_posteriors, rgbView_parzen_posteriors)


        # Ensemble classifiers using the sum rule
        # predicted_class = predicaoCombinada(prioris, posteriors)

        #print("actual:", classe_correta, " prediction:", predicted_class)
        #print("true class: %s" % classe_correta)
        #print("predicted: %s"% predicted_class)

        # usado para gerar matriz de confusao
        prediction = []
        prediction.append(classe_correta)
        prediction.append(predicted_class)
        predictions.append(prediction)
        # print(classe_correta, predicted_class,
        #       classe_correta == predicted_class)

        # para calculo de taxa de erro
        if (classe_correta == predicted_class):
            true_positives += 1
        else:
            false_positives += 1
        estimativas.append(predicted_class)

    error_rate = false_positives / (true_positives + false_positives)

    error_rates.append(error_rate)



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

for indice, a in enumerate(acuracias):
    print(indice, ': %.5f' % (a))
resultados['RegraDaSoma'] = acuracias

print("\tCOMBINADO: Confusion Matrix ####################")
predictions = np.array(predictions)
confusion_matrix = ConfusionMatrix(predictions[:, 0], predictions[:, 1])
print(confusion_matrix)
confusion_matrix.to_dataframe().to_csv(
    "results/regrasoma_matriz_confusao.csv")
# confusion_matrix.print_stats()
print("")

print("\tCOMBINADO: Precision by Class")
precision_by_class = confusion_matrix.to_dataframe().values.diagonal() / float(
    elementsByClass * repeticoes)
for index, classe in enumerate(nomeClasses):
    print(classe, ' : %.5f' % (precision_by_class[index]))
print("")


print("\tCOMPLETE VIEW: precision average %s" % np.mean(precision_by_class))
print("\tCOMPLETE VIEW: error rate average %s" % np.mean(error_rates))

print("")

endTotalTime = time.time()
print("Total        : %.2f segundos" % (endTotalTime - startTotalTime))
pd.DataFrame(resultados).to_csv("results/regrasoma_acuracias.csv")