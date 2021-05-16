import numpy as np
import logging
import datetime

from typing import Optional
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger('k_mxt_w.clusters_data')


class ClustersData(ABC):
    """
    Base class of clusters data
    """
    def __init__(self):
        self.cluster_numbers = None
        self.num_of_data = 0
        self.data_ration = None

    @staticmethod
    def array_rationing(array: np.ndarray) -> np.ndarray:
        """
        normalizes the array values
        :param array: array which will be normalized
        :return: normalazed array
        """
        rationed_array = (array - np.mean(array)) / np.std(array)
        if np.isnan(rationed_array).any():
            logger.info(f'It is impossible to ration array={array}.')
            return array
        return rationed_array

    @abstractmethod
    def distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
        """
        calculates distance in array data_ration between point1 and point2.
        If point2 is equal to None, the function calculates distance between point1 and other points of data_ration array
        :param point1: the point1 number in data_ration array
        :param point2: the point2 number in data_ration array
        :return: distance between point1 and point2 or between point1 and others
        """
        raise NotImplementedError

    def get_cluster_name(self, cluster_num: int):
        """
        Return string representation of cluster number
        :param cluster_num: cluster number
        :return: string representation of cluster number
        """
        if self.cluster_numbers is None:
            raise TypeError('self.cluster_numbers cannot be equal to None')
        if cluster_num in self.cluster_numbers:
            return str(cluster_num)
        raise ValueError(f'cluster_num = {cluster_num} is not correct cluster number')

    def calculate_modularity(self, graph):
        """
        Calculates modularity metric
        :param graph: graph on which will be calculated the metric
        :return: value of modularity metric
        """
        if self.cluster_numbers is None:
            raise TypeError('self.cluster_numbers cannot be equal to None')
        modularity_value = 0
        edge_count = sum([len(graph[i]) for i in range(len(graph))])
        cnt_edge_in_cluster = defaultdict(int)
        cnt_edge_connecting_with_cluster = defaultdict(int)
        for v in range(len(graph)):
            for to in graph[v]:
                if self.cluster_numbers[v] == self.cluster_numbers[to]:
                    cnt_edge_in_cluster[self.cluster_numbers[v]] = \
                        cnt_edge_in_cluster.get(self.cluster_numbers[v], 0) + 1
                cnt_edge_connecting_with_cluster[self.cluster_numbers[v]] = \
                    cnt_edge_connecting_with_cluster.get(self.cluster_numbers[v], 0) + 1
        for current_cluster in set(self.cluster_numbers):
            modularity_value += cnt_edge_in_cluster[current_cluster] / edge_count \
                                - ((cnt_edge_connecting_with_cluster[current_cluster] / edge_count) ** 2)
        return modularity_value


class MetricsMixin:
    """
    Contains metric which calculated distance between points
    """
    def __init__(self):
        self.data_ration = None
        self.time_init = None

    @staticmethod
    def __calculate_distance(point1, point2, func_for_one_point, func_for_two_points):
        """
        If point1 and point2 is int, then func_for_two_points will be called (case 1)
        If point2 is equal to None, then func_for_one_point will be called (case 2)
        :param point1: the point1 number in data_ration array
        :param point2: the point2 number in data_ration array
        :param func_for_one_point: function which will be used in the second case
        :param func_for_two_points: function which will be used in the first case
        :return: result of func_for_one_point or func_for_two_points
        """
        if point2 is None:
            return func_for_one_point(num_point=point1)
        else:
            return func_for_two_points(point1=point1, point2=point2)

    def euclidean_distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
        """
        Calculates Euclidean distance in array data_ration between point1 and point2.
        If point2 is equal to None, the function calculates Euclidean distance between point1
        and other points of data_ration array
        :param point1: the point1 number in data_ration array
        :param point2: the point2 number in data_ration array
        :return: Euclidean distance between point1 and point2 or between point1 and others
        """
        def euclidean_distance_between_point_array(num_point: int):
            return np.sqrt(np.sum((self.data_ration - self.data_ration[num_point]) ** 2, axis=1))

        def euclidean_distance_between2points(point1, point2):
            return np.sqrt(np.sum((self.data_ration[point1] - self.data_ration[point2]) ** 2))

        return MetricsMixin.__calculate_distance(
            point1,
            point2,
            func_for_one_point=euclidean_distance_between_point_array,
            func_for_two_points=euclidean_distance_between2points,
        )

    def manhattan_distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
        """
        Calculates Manhattan distance in array data_ration between point1 and point2.
        If point2 is equal to None, the function calculates Manhattan distance between point1
        and other points of data_ration array
        :param point1: the point1 number in data_ration array
        :param point2: the point2 number in data_ration array
        :return: Manhattan distance between point1 and point2 or between point1 and others
        """
        def manhattan_distance_between_point_array(num_point: int):
            return np.abs(np.sum(self.data_ration - self.data_ration[num_point], axis=1))

        def manhattan_distance_between2points(point1, point2):
            return np.abs(np.sum(self.data_ration[point1] - self.data_ration[point2]))

        return MetricsMixin.__calculate_distance(
            point1,
            point2,
            func_for_one_point=manhattan_distance_between_point_array,
            func_for_two_points=manhattan_distance_between2points,
        )


class ClustersDataSpace(ClustersData, MetricsMixin, ABC):
    """
    Contains clustering result, metric, coordinates of points etc.
    """
    def __init__(self, x_init: np.ndarray, y_init: np.ndarray, metrics: str):
        """

        :param x_init: contains coordinates of point along x-axis
        :param y_init: contains coordinates of point along y-axis
        :param metrics: contains the name of metric which will be used to calculate distance between points
        """
        super().__init__()
        if x_init.shape != y_init.shape:
            raise ValueError('x_init and y_init must have the same dimension')
        self.x_init = x_init.copy()
        self.y_init = y_init.copy()
        self.data_ration = None
        self.cluster_numbers = np.full(len(self.x_init), -1)
        self.num_of_data = self.x_init.shape[0]
        self.__allowed_metrics = {
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
        }
        self.metrics = metrics
        self._distance_func = self.__allowed_metrics.get(self.metrics)
        if self._distance_func is None:
            raise ValueError(f'metrics={metrics} is not correct value. '
                             f'{self.__allowed_metrics.keys()} is the list of possibles values.')

    def distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
        """
        Calculates distance in array data_ration between point1 and point2.
        If point2 is equal to None, the function calculates distance between point1 and other points of data_ration array
        :param point1: the point1 number in data_ration array
        :param point2: the point2 number in data_ration array
        :return: distance between point1 and point2 or between point1 and others
        """
        return self._distance_func(point1, point2)


class ClustersDataSpace2d(ClustersDataSpace):
    """
    Contains information about clusters of data in 2d like clustering result, metric, coordinates of points etc.
    """
    def __init__(self, x_init: np.ndarray, y_init: np.ndarray, metrics='euclidean'):
        """
        :param x_init: contains coordinates of point along x-axis
        :param y_init: contains coordinates of point along y-axis
        :param metrics: contains the name of metric which will be used to calculate distance between points
        """
        ClustersDataSpace.__init__(self, x_init, y_init, metrics=metrics)
        self.data_ration = np.array([self.x_init,
                                     self.y_init]).transpose()


class ClustersDataSpaceFeatures(ClustersDataSpace):
    """
    Contains information about clusters of data in multidimensional space
    like clustering result, metric, coordinates of points etc.
    """
    def __init__(self, x_init: np.ndarray, y_init: np.ndarray, features_init: np.ndarray, metrics='euclidean'):
        """
        :param x_init: contains coordinates of point along x-axis
        :param y_init: contains coordinates of point along y-axis
        :param metrics: contains the name of metric which will be used to calculate distance between points
        :param features_init: each row corresponds to a single data point.
        """
        super().__init__(x_init, y_init, metrics=metrics)
        self.features_init = features_init.copy()
        self.data_ration = np.concatenate((ClustersData.array_rationing(self.x_init.reshape(-1, 1)),
                                           ClustersData.array_rationing(self.y_init.reshape(-1, 1)),
                                           ClustersData.array_rationing(self.features_init)), axis=1)
