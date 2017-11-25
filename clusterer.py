import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import squareform, pdist
from graph import Graph

num_cluster = 15
training_fraction = 0.01

data_source = "CensusIncome\CencusIncome.data.txt"
full_header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
numeric_header = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
statistic_file = 'data_stat.pkl'


def read_numeric_training_data():
    training_data = pd.read_csv(data_source, header=None, index_col=False, names=full_header)
    training_data.replace('?', np.NaN)
    training_data = training_data[numeric_header + ["class"]]
    training_data = training_data.drop_duplicates(subset=numeric_header, keep='first')
    training_data.dropna(axis=0, how='any')
    return training_data


def standardize_data(training_data, means=[], sdts=[]):
    if not means:
        means = training_data.mean()
    if not sdts:
        sdts = training_data.std()
    for coll in numeric_header:
        training_data[coll] = (training_data[coll] - means[coll]) / sdts[coll]
    return training_data, means, sdts


def generate_dist_matrix(training_data):
    dist_matrix = pd.DataFrame(squareform(pdist(training_data[numeric_header].iloc[:, 1:])))
    return dist_matrix


def generate_dist_tree(dist_matrix):
    g = Graph(len(dist_matrix))
    for i in range(len(dist_matrix)):
        for j in range(i+1, len(dist_matrix)):
            g.set_dist(i,j,dist_matrix[i][j])
    g.gen_min_spanning_tree()
    return g


def clusterize_data(training_data):
    dist_matrix = generate_dist_matrix(training_data)
    dist_tree = generate_dist_tree(dist_matrix)
    dist_tree.cut_longest_conn(num_cluster-1)

    clusters = dist_tree.get_connected_tree()
    cluster_data = []
    for cluster in clusters:
        cluster_data.append(training_data.iloc[list(cluster)])
    return cluster_data


def get_purity(data_list):
    purities = []
    for df in data_list:
        mode = df["class"].mode().iloc[0]
        purities.append(len(df.loc[df['class'] == mode]) / len(df))
    return purities


def save_data(data):
    with open(statistic_file, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


def load_data():
    with open(statistic_file, 'rb') as input:
        data = pickle.load(input)
        return data


def main():
    data = read_numeric_training_data()
    data = data.sample(frac=training_fraction, random_state=17)
    data, means, sdts = standardize_data(data)

    data = clusterize_data(data)

    print(get_purity(data))

if __name__ == "__main__":
    main()
