import numpy as np


def calculateConfusionMatrix(repetitions_predictions):
    matrix = np.zeros((7, 7))
    for predictions in repetitions_predictions:
        for prediction in predictions:
            matrix[prediction[0], prediction[1]] += 1
    return matrix


# calculate prior for classes
def calculatePrior(train_set):
    qtdElementos = 0
    for conjunto in train_set:
        qtdElementos += len(train_set[conjunto])

    probabilidadesPriori = {}
    for classe, elementos in train_set.items():
        probabilidadesPriori[classe] = len(elementos) / qtdElementos
    return probabilidadesPriori