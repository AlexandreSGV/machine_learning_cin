import pandas as pd
import numpy as np


import csv

my_dict = {"test": [1,2,3,4], "testing": [5,6,7,8]}

df = pd.DataFrame(my_dict)
df.to_csv("file_path.csv")
