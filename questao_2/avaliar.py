import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('results/total_acuracias.csv', sep=',')

for column in df:
    res_mean, res_var, res_std = stats.bayes_mvs(df[column], alpha=0.95)
    print(column,res_mean)

measurements = df.values
test_stat, p_value = stats.friedmanchisquare(*measurements)
print(p_value)


raise Exception()
