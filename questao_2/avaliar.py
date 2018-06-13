import Orange as orange
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

dataframe = pd.read_csv('results/total_acuracias.csv', sep=',')

for column in dataframe:
    res_mean, res_var, res_std = stats.bayes_mvs(dataframe[column], alpha=0.95)
    print(column, res_mean, res_var, res_std)

dados = np.asarray(dataframe.values)
dadosT = dados.T
test_stat, p_value = stats.friedmanchisquare(*dadosT)
print(p_value)
print('stat', test_stat)
print(dataframe.describe())

colunas = [
    'Gaussian\nComplete', 'Gaussian RGB', 'Gaussian Shape',
    'Parzen Complete', 'Parzen RGB', 'Parzen Shape', 'Regra da Soma'
]

ranking_de_medias = dataframe.rank(axis=1, ascending=False)
ranking_medias = ranking_de_medias.mean()
distancia_critica = orange.evaluation.compute_CD(
    avranks=ranking_medias, n=30, alpha='0.05', test='nemenyi')
orange.evaluation.graph_ranks(
    avranks=ranking_medias, names=colunas, cd=distancia_critica)


plt.show()
print('distancia_critica', distancia_critica)
# print(ranking_de_medias.columns)
# plt.savefig('')
