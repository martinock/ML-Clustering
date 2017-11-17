import pandas as pd
from graph import Graph


class Clusterer:

    def __init__(self):
        self.data = pd.read_csv("CensusIncome\CencusIncome.data.txt", header=None,
                                names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"])
        print(self.data)

def main():
    c = Clusterer()

if __name__ == "__main__":
    main()
