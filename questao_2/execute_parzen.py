import pandas as pd
import numpy as np
from common_lib import *
from parzem_lib import *
import time
startTotalTime = time.time()
# startTotalTime = time.time()
repeticoes = 30
splits = 10
elementsByClass = 300

# df = pd.read_csv('image_segmentation_18_2098.csv', sep=';')
df = pd.read_csv('data/segmentation_18_col.csv', sep=',')
# print('colluns', df.columns)
df = df.sort_values('CLASSE')

# raise Exception()

nomeClasses = df.CLASSE.unique()
print(nomeClasses)
numeroClasses = len(nomeClasses)

# dados = df.values
# dados = df.iloc[:, 1:].values
gabarito = df.CLASSE.values

# ##############################################################
# Complete View
startCompleteViewTime = time.time()
print('#################### COMPLETE VIEW ####################')
dadosCompleteView = df.iloc[:, 1:].values
print('COMPLETE Columns', df.iloc[:, 1:].columns)

predictions, error_rates, acuracias = executaParzen(
    dadosCompleteView, gabarito, nomeClasses, repeticoes, splits,
    elementsByClass)
for indice, a in enumerate(acuracias):
    print(indice, ': %.5f' % (a))

print("\tCOMPLETE VIEW: Confusion Matrix ####################")
confusionMatrix = calculateConfusionMatrix(predictions)
print(np.array_str(confusionMatrix, precision=6, suppress_small=True))
print("")

print("\tCOMPLETE VIEW: Precision by Class")
precision_by_class = confusionMatrix.diagonal() / float(
    elementsByClass * repeticoes)
print(precision_by_class)
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
print('Shape Columns', df.iloc[:, 1:9].columns)

predictions, error_rates, acuracias = executaParzen(
    dadosShapeView, gabarito, nomeClasses, repeticoes, splits,
    elementsByClass)
for indice, a in enumerate(acuracias):
    print(indice, ': %.5f' % (a))

print("\tSHAPE VIEW: Confusion Matrix ####################")
confusionMatrix = calculateConfusionMatrix(predictions)
print(np.array_str(confusionMatrix, precision=6, suppress_small=True))
print("")

print("\tSHAPE VIEW: Precision by Class")
precision_by_class = confusionMatrix.diagonal() / float(
    elementsByClass * repeticoes)
print(precision_by_class)
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
print('RGB Columns', df.iloc[:, 9:19].columns)
predictions, error_rates, acuracias = executaParzen(
    dadosRGBView, gabarito, nomeClasses, repeticoes, splits,
    elementsByClass)
for indice, a in enumerate(acuracias):
    print(indice, ': %.5f' % (a))

print("\tRGB VIEW: Confusion Matrix ####################")
confusionMatrix = calculateConfusionMatrix(predictions)
print(np.array_str(confusionMatrix, precision=6, suppress_small=True))
print("")

print("\tRGB VIEW: Precision by Class")
precision_by_class = confusionMatrix.diagonal() / float(
    elementsByClass * repeticoes)
print(precision_by_class)
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