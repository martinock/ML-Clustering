import math
import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import squareform, pdist
from graph import Graph

num_cluster = 50
training_fraction = 0.5

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


def standardize_data(training_data, means=None, sdts=None):
    if means is None:
        means = training_data.mean()
    if sdts is None:
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


def clusterize_data(training_data, dist_tree = None):
    if dist_tree is None:
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


def generate_centroids(data_clusters):
    centroids = []
    for cluster in data_clusters:
        centroid = cluster.mean();
        centroid.ix['class'] = cluster["class"].mode().iloc[0]
        centroids.append(centroid)
    return centroids


def predict_class(centroids, test_row):
    best_centroid, best_dist = None, math.inf
    for centroid in centroids:

        curr_dist = 0
        for header in numeric_header:
            curr_dist += (centroid[header] - test_row[header])**2

        curr_dist = math.sqrt(curr_dist)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_centroid = centroid

    return best_centroid["class"]


def get_accuracy(centroids, test_data):
    correct_count = 0
    for index, row in test_data.iterrows():
        if predict_class(centroids, row) == row["class"]:
            correct_count += 1
    return correct_count / len(test_data)


def save_data(data):
    with open(statistic_file, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


def load_data():
    with open(statistic_file, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p


def main():
    data = read_numeric_training_data()
    training_data = data.sample(frac=training_fraction, random_state=17)
    training_data, means, sdts = standardize_data(training_data)

    dist_tree = load_data()
    data_cluster = clusterize_data(training_data, dist_tree)

    print(get_purity(data_cluster))
    centroids = generate_centroids(data_cluster)

    test_data, _, _ = standardize_data(data, means, sdts)
    print(get_accuracy(centroids, test_data))


if __name__ == "__main__":
    main()
