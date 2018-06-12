import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score

dados = pd.read_csv('image_segmentation_16.csv', sep=';')
Y = dados.iloc[:, 0:1].values # conjunto de dados
print(Y)

labels_true = []
for j in range(len(Y)):
    for h in range(len(Y[j])):
        print (Y[j][h])
# labels_true = ['grass', 'brick', 'sky','grass', 'brick', 'sky','grass', 'brick', 'sky']
# labels_pred = [1, 2, 3, 1, 2, 3, 1, 2, 3]

# print(adjusted_rand_score(labels_true, labels_pred) )