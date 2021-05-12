from abc import ABC
import numpy as np
import copy
import graph
import logging
import scipy.stats

import clusters_data

logger = logging.getLogger('k_mxt_w.clustering_algorithm')


class Clustering(ABC):
    """
    The base class for clustering algorithms
    """
    def __init__(self, clusters_data: clusters_data.ClustersData):
        """
        :param clusters_data: object of class clusters_data.ClustersData
        """
        self.clustersData = copy.deepcopy(clusters_data)


class K_MXT(Clustering):
    """
    Makes clustering using k-mxt algorithm
    """
    def __init__(self, k: int, eps: float, clusters_data: clusters_data.ClustersData):
        """
        :param k: the maximum possible number of incidenting arcs  from each vertex
        :param eps: distance in which will be connected vertices of the graph
        :param clusters_data: object of class clusters_data.ClustersData,
        which contains clustering result, metric, coordinates of points etc.
        """
        logger.info(f'init k-{k}, eps-{eps}, clusters_data-{clusters_data}')
        self.k = k
        self.eps = eps
        self.clusters_data = clusters_data
        self.num_of_vertices = self.clusters_data.num_of_data
        self.start_graph = [None for _ in range(self.num_of_vertices)]
        self.k_graph = [None for _ in range(self.num_of_vertices)]
        logger.info(f'init has done.')

    def make_start_graph(self):
        """
        Builds a started undirected graph, where vertices are points on the plane.
        The edges of graph are the connections between the vertices located at a distance less than eps.
        :return: None
        """
        def get_neighbor(v):
            """
            Finds neighbors for vertex v
            :param v: the vertex number
            :return: None
            """
            dst = self.clusters_data.distance(v)
            neighbor = np.where(dst <= self.eps)[0]
            index_v = np.argwhere(neighbor == v)
            self.start_graph[v] = np.delete(neighbor, index_v)

        for v in range(self.clusters_data.num_of_data):
            get_neighbor(v)

    def get_arc_weight(self, v, to):
        """
        Calculates weight of arc between v and to vertices
        :param v: vertex 1
        :param to: vertex 2
        :return:  weight of arc (v, to)
        """
        return np.intersect1d(self.start_graph[v], self.start_graph[to]).shape[0]

    def make_k_graph(self):
        """
        According to the started graph the function builds an oriented subgraph with the same set of vertices.
        For each vertex it adds k its incident edges with the greates weight to the subgraph as arcs.
        If it is necessary to choose from several edges with the same weight, the selection is carried out randomly.
        :return: None
        """
        if any(x is None for x in self.start_graph):
            raise TypeError('self.start_graph do not have to consist None.')
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        np.random.seed(4000)
        edge_weights = None

        def get_k_max_arcs():
            """
            Finds and returns k arcs which have maximum weights
            :return: the list of vertices which have maximum weights
            """
            if edge_weights.shape[0] < self.k:
                return edge_weights['vertex']
            edge_weights.sort(order='weight')
            weights = np.flip(edge_weights)
            k_value = weights['weight'][self.k - 1]
            index_k_value = np.where(weights['weight'] == k_value)[0]
            index_k_value_min, index_k_value_max = index_k_value[0], index_k_value[-1]
            np.random.shuffle(weights[index_k_value_min:index_k_value_max+1])
            return weights['vertex'][:self.k]

        for v in range(self.num_of_vertices):
            iterable = ((self.get_arc_weight(v, neighbor), neighbor) for neighbor in self.start_graph[v])
            edge_weights = np.fromiter(iterable, dtype=[('weight', float), ('vertex', int)])
            self.k_graph[v] = get_k_max_arcs()
            edge_weights = None

    def __call__(self, *args, **kwargs):
        logger.info(f'clustering has started')
        logger.info(f'start making start graph')
        self.make_start_graph()
        logger.info(f'start making k-graph')
        self.make_k_graph()
        logger.info(f'start finding scc')
        g = graph.Graph(adj=self.k_graph)
        self.clusters_data.cluster_numbers = np.array(g.find_scc())
        logger.info(f'clustering has finished')


class K_MXT_gauss(K_MXT):
    """
    Makes clustering using k-mxt-gauss algorithm
    """
    def __init__(self, k: int, eps: float, clusters_data: clusters_data.ClustersData):
        """
        :param k: the maximum possible number of incidenting arcs  from each vertex
        :param eps: distance in which will be connected vertices of the graph
        :param clusters_data: object of class clusters_data.ClustersData,
        which contains clustering result, metric, coordinates of points etc.
        """
        super().__init__(k=k, eps=eps, clusters_data=clusters_data)
        self.sigma = eps / 3
        self.norm = scipy.stats.norm(0, self.sigma)

    def get_arc_weight(self, v, to):
        """
        Calculates weight of arc between v and to vertices
        :param v: vertex 1
        :param to: vertex 2
        :return:  weight of arc (v, to)
        """
        return self.norm.pdf(self.clusters_data.distance(v, to)) * super().get_arc_weight(v, to)




