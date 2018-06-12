import Orange as orange
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

dataframe = pd.read_csv('results/total_acuracias.csv', sep=',')

for column in dataframe:
    res_mean, res_var, res_std = stats.bayes_mvs(dataframe[column], alpha=0.95)
    print(column,res_mean)

dados = np.asarray(dataframe.values)
dadosT = dados.T
test_stat, p_value = stats.friedmanchisquare(*dadosT)
print(p_value)


ranking_medias = dataframe.rank(axis=1, ascending=False)
ranking_medias = ranking_medias.mean()
distancia_critica = orange.evaluation.compute_CD(
    avranks=ranking_medias, n=len(dataframe), alpha='0.05', test='nemenyi')
orange.evaluation.graph_ranks(
    avranks=ranking_medias, names=dataframe.columns, cd=distancia_critica)
plt.show()
# plt.savefig('')
