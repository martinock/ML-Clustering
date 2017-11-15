import networkx as nx
import matplotlib.pyplot as plt


class Graph:

    def __init__(self, num_data):
        self.wg = nx.complete_graph(num_data)
        for n, nadjs in self.wg.adj.items():
            for nadj, eattrs in nadjs.items():
                self.wg[n][nadj]['weight'] = 0.0
        self.g = None

    def dist_str(self):
        out_str = ''
        for n, nadjs in self.wg.adj.items():
            for nadj, eattrs in nadjs.items():
                out_str += '(%s,%s):%3.2f\t' % (n, nadj, eattrs['weight'])
            out_str += "\n"
        return out_str

    def set_dist(self, i,j, distance):
        self.wg.edges[i, j]['weight'] = distance

    def gen_min_spanning_tree(self):
        self.g = nx.minimum_spanning_tree(self.wg)

    def cut_longest_conn(self, num=0):
        if num < 1:
            return
        if self.g.isNone():
            self.gen_min_spanning_tree()
        max, i, j = 0.0, 0, 0
        for n, nadjs in self.g.adj.items():
            for nadj, eattrs in nadjs.items():
                if max < eattrs['weight']:
                    max, i, j = eattrs['weight'], n, nadj
        self.g.remove_node(i, j)
        self.cut_longest_conn(num-1)

    def get_connected_tree(self):
        if self.g.isNone():
            self.gen_min_spanning_tree()
        return nx.connected_components(self.g)


def main():
    g = Graph(5)
    print(g.dist_str())

if __name__ == "__main__":
    main()
