import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import squareform, pdist
from pathlib import Path
from graph import Graph


class Clusterer:

    data_source = "CensusIncome\CencusIncome.data.txt"
    full_header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
    numeric_header = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    statistic_file = 'data_stat.pkl'

    @staticmethod
    def read_numeric_training_data():
        training_data = pd.read_csv(Clusterer.data_source, header=None, index_col=False, names=Clusterer.full_header)
        training_data.replace('?', np.NaN)
        training_data = training_data[Clusterer.numeric_header + ["class"]]
        training_data = training_data.drop_duplicates(subset=Clusterer.numeric_header, keep='first')
        training_data.dropna(axis=0, how='any')
        return training_data

    def __init__(self, isload):
        '''
        Initiate the data means, standard deviation, and distance matrix
        :param isload: Whether or not generate the statistic from scratch or to load it from the data
        '''
        if not isload:
            training_data = Clusterer.read_numeric_training_data()
            self.means = training_data.mean()
            self.sdts = training_data.std()
            for coll in Clusterer.numeric_header:
                training_data[coll] = (training_data[coll] - self.means[coll]) / self.sdts[coll]
            self.dist_matrix = pd.DataFrame(squareform(pdist(training_data[Clusterer.numeric_header].iloc[:, 1:])))
            self.save_stat()
        else:
            self.load_stat()

    def load_stat(self):
        with open(self.statistic_file, 'rb') as input:
            data = pickle.load(input)
            self.means = data[0]
            self.sdts = data[1]
            self.dist_matrix = data[2]

    def save_stat(self):
        with open(self.statistic_file, 'wb') as output:
            pickle.dump([self.means, self.sdts, self.dist_matrix], output, pickle.HIGHEST_PROTOCOL)


def main():
    #c = Clusterer(False)
    c = Clusterer(True)

if __name__ == "__main__":
    main()
