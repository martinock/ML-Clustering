import pandas as pd
import numpy as np
from graph import Graph


class Clusterer:

    data_source = "CensusIncome\CencusIncome.data.txt"
    full_header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
    numeric_header = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    def __init__(self):
        self.training_data = pd.read_csv(self.data_source, header=None, index_col=False, names=self.full_header)
        self.training_data.replace('?', np.NaN)
        self.training_data = self.training_data[self.numeric_header + ["class"]]
        self.training_data = self.training_data.drop_duplicates(subset=self.numeric_header, keep='first')
        self.training_data.dropna(axis=0, how='any')

        means = self.training_data.mean()
        sdts = self.training_data.std()
        for coll in self.numeric_header:
            self.training_data[coll] = (self.training_data[coll] - means[coll]) / sdts[coll]

        print(self.training_data)


def main():
    c = Clusterer()

if __name__ == "__main__":
    main()
