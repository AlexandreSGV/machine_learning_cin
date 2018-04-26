import random

def random_class(dataframe, class_name):
    return (dataframe.loc[dataframe['classe'] == class_name]).sample(n=1)