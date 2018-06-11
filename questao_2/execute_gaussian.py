import pandas as pd
from gaussian import *
from common import *
import numpy as np
import codigo_modulado as code
import time
# from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from pandas_ml import ConfusionMatrix

# y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
# y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

# confusion_matrix = ConfusionMatrix(y_true, y_pred)
# print("Confusion matrix:\n%s" % confusion_matrix)

# raise Exception()
startTotalTime = time.time()
repeticoes = 30
splits = 10
elementsByClass = 300


df = pd.read_csv('data/segmentation_18_col.csv', sep=',')
nomeClasses = df.CLASSE.unique()
print (nomeClasses)
gabarito = df.CLASSE.values

# variável armazenara os resultados em cada viev (col: view x row: repetition)
resultados = {}

# ##############################################################
# Complete View
startCompleteViewTime = time.time()
print('#################### COMPLETE VIEW ####################')
dadosCompleteView = df.iloc[:, 1:].values
predictions, error_rates, acuracias = executaGaussiana(
    dadosCompleteView, gabarito, nomeClasses, repeticoes, splits)
for indice, a in enumerate(acuracias):
    print(indice, ': %.5f' % (a))
resultados['GaussianCompleteView'] = acuracias

print("\tCOMPLETE VIEW: Confusion Matrix ####################")
predictions = np.array(predictions)
confusion_matrix = ConfusionMatrix(predictions[:, 0], predictions[:, 1])
print(confusion_matrix)
confusion_matrix.to_dataframe().to_csv("results/gaussian_matriz_confusao_complete_view.csv")
# confusion_matrix.print_stats()
print("")

print("\tCOMPLETE VIEW: Precision by Class")
precision_by_class = confusion_matrix.to_dataframe().values.diagonal() / float(
elementsByClass * repeticoes)
for index, classe in enumerate(nomeClasses):
    print(classe, ' : %.5f' % (precision_by_class[index]))
print("")


print("\tCOMPLETE VIEW: precision average %s" % np.mean(precision_by_class))
print("\tCOMPLETE VIEW: error rate average %s" % np.mean(error_rates))
print("")

endCompleteViewTime = time.time()

# ##############################################################
# Shape View
startShapeViewTime = time.time()
print('#################### SHAPE VIEW ####################')
dadosShapeView = df.iloc[:, 1:9].values
predictions, error_rates, acuracias = executaGaussiana(
    dadosShapeView, gabarito, nomeClasses, repeticoes, splits)
for indice, a in enumerate(acuracias):
    print(indice, ': %.5f' %(a) )
resultados['GaussianShapeView'] = acuracias

print("\tSHAPE VIEW: Confusion Matrix ####################")
predictions = np.array(predictions)
confusion_matrix = ConfusionMatrix(predictions[:, 0], predictions[:, 1])
print(confusion_matrix)
confusion_matrix.to_dataframe().to_csv(
    "results/gaussian_matriz_confusao_shape_view.csv")
# confusion_matrix.print_stats()
print("")

print("\tSHAPE VIEW: Precision by Class")
precision_by_class = confusion_matrix.to_dataframe().values.diagonal() / float(
    elementsByClass * repeticoes)
for index, classe in enumerate(nomeClasses):
    print(classe, ' : %.5f' % (precision_by_class[index]))
print("")

print("\tSHAPE VIEW: precision average %s" % np.mean(precision_by_class))
print("\tSHAPE VIEW: error rate average %s" % np.mean(error_rates))
print("")

endShapeViewTime = time.time()

# ##############################################################
# RGB View
startRGBViewTime = time.time()
print('#################### RGB VIEW ####################')
dadosRGBView = df.iloc[:, 9:19].values
predictions, error_rates, acuracias = executaGaussiana(
    dadosRGBView, gabarito, nomeClasses, repeticoes, splits)
for indice, a in enumerate(acuracias):
    print(indice, ': %.5f' % (a))
resultados['GaussianRGBView'] = acuracias

print("\tRGB VIEW: Confusion Matrix ####################")
predictions = np.array(predictions)
confusion_matrix = ConfusionMatrix(predictions[:, 0], predictions[:, 1])
print(confusion_matrix)
confusion_matrix.to_dataframe().to_csv(
    "results/gaussian_matriz_confusao_rgb_view.csv")
# confusion_matrix.print_stats()
print("")

print("\tRGB VIEW: Precision by Class")
precision_by_class = confusion_matrix.to_dataframe().values.diagonal() / float(
    elementsByClass * repeticoes)
for index, classe in enumerate(nomeClasses):
    print(classe, ' : %.5f' % (precision_by_class[index]))
print("")

print("\tRGB VIEW: precision average %s" % np.mean(precision_by_class))
print("\tRGB VIEW: error rate average %s" % np.mean(error_rates))
print("")

endRGBViewTime = time.time()


endTotalTime = time.time()

print('TEMPOS DE EXECUÇÃO ')
print("CompleteView : %.2f segundos" %
      (endCompleteViewTime - startCompleteViewTime))
print("ShapeView    : %.2f segundos" % (endShapeViewTime - startShapeViewTime))
print("RGBView      : %.2f segundos" % (endRGBViewTime - startRGBViewTime))
print("Total        : %.2f segundos" % (endTotalTime - startTotalTime))

pd.DataFrame(resultados).to_csv("results/gaussian_acuracias.csv")