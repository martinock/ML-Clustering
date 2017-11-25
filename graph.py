import networkx as nx


class Graph:

    def __init__(self, num_data):
        """
        Initiate the data graph with as many node as the number of data
        :param num_data: The number of data to be processed
        """
        self.data_graph = nx.complete_graph(num_data)
        for n, nadjs in self.data_graph.adj.items():
            for nadj, eattrs in nadjs.items():
                self.data_graph[n][nadj]['weight'] = 0.0

    def __str__(self):
        """
        :return: String representation of the distance between data nodes
        """
        out_str = ''
        for n, nadjs in self.data_graph.adj.items():
            for nadj, eattrs in nadjs.items():
                out_str += '(%s,%s):%3.2f\t' % (n, nadj, eattrs['weight'])
            out_str += "\n"
        return out_str

    def set_dist(self, i,j, distance):
        """
        Set the distance between two data nodes
        :param i: The first data node id (row number)
        :param j: The second data node id (row number)
        :param distance: The distance between the nodes in Float
        """
        if self.data_graph.has_edge(i,j):
            self.data_graph.edges[i, j]['weight'] = distance

    def gen_min_spanning_tree(self):
        """
        Calculate the spanning tree of the data graph, and save the resulting tree in tree graph
        """
        # TODO: Create own implementation of this
        self.data_graph = nx.minimum_spanning_tree(self.data_graph)

    def cut_longest_conn(self, num=0):
        """
        Recursively cut the edges with the longest distance in the tree graph, generate the tree graph if not generated
        :param num: The number of connection to be cut
        """
        maxs, edges = [], []
        for _ in range(num):
            maxs.append(0.0); edges.append((-1,-1));

        for n, nadjs in self.data_graph.adj.items():
            for nadj, eattrs in nadjs.items():

                if min(maxs) < eattrs['weight']:
                    if (nadj, n) not in edges:
                        idx = maxs.index(min(maxs))
                        maxs[idx], edges[idx]= eattrs['weight'], (n, nadj)

        self.data_graph.remove_edges_from(edges)

    def get_connected_tree(self):
        """
        :return: A set of set of connected nodes in the tree graph
        """
        if self.data_graph == None:
            self.gen_min_spanning_tree()
        return [n for n in nx.connected_components(self.data_graph)]


def main():
    g = Graph(1000)

if __name__ == "__main__":
    main()
