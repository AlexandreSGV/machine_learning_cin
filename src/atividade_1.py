import pandas as pd  
import numpy as np 
from pandas import Series, DataFrame
import math
from utils import utils
from algorithms.kcm import KCM

dataset = pd.read_csv('dataset/segmentation.test')
partition_number = 7

print(dataset.classe.unique())

kcm = KCM(dataset, partition_number)
kcm.run()
