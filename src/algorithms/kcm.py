from utils import utils

class KCM:
    def __init__(self, dataframe, clusters_number):
        self.dataframe = dataframe
        self.clusters_number = clusters_number
        self.classes = self.dataframe.classe.unique()
    
    def initializion(self):
        # passo 1: selecionar aleatoriamente 'c' (um elemento de cada classe do dataframe)
        lista = [utils.random_class(self.dataframe, c) for c in self.classes]

    def run(self):
        self.initializion()
        pass