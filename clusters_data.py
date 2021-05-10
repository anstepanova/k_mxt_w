import numpy as np
import logging
import datetime

from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger('k_mxt_w.clusters_data')


class ClustersData(ABC):
    def __init__(self):
        self.cluster_numbers = None
        self.num_of_data = 0
        self.data_ration = None

    @staticmethod
    def array_rationing(array: np.ndarray) -> np.ndarray:
        rationed_array = (array - np.mean(array)) / np.std(array)
        if np.isnan(rationed_array).any():
            logger.info(f'It is impossible to ration array={array}.')
            return array
        return rationed_array

    @abstractmethod
    def distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
        raise NotImplementedError

    def get_cluster_name(self, cluster_num: int) -> str:
        if self.cluster_numbers is None:
            raise TypeError('self.cluster_numbers cannot be equal to None')
        return str(self.cluster_numbers[cluster_num])


class MetricsMixin:
    def __init__(self):
        self.data_ration = None
        self.time_init = None

    @staticmethod
    def __calculate_distance(point1, point2, func_for_one_point, func_for_two_points):
        if point2 is None:
            return func_for_one_point(num_point=point1)
        else:
            return func_for_two_points(point1=point1, point2=point2)

    def euclidean_distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
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

    def minkowski_distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
        raise NotImplementedError


class ClustersDataSpace(ClustersData, MetricsMixin, ABC):
    def __init__(self, x_init: np.ndarray, y_init: np.ndarray, metrics: str):
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
            'minkowski': self.minkowski_distance,
        }
        self.metrics = metrics
        self._distance_func = self.__allowed_metrics.get(self.metrics)
        if self._distance_func is None:
            raise ValueError(f'metrics={metrics} is not correct value. '
                             f'{self.__allowed_metrics.keys()} is the list of possibles values.')

    def distance(self, point1: np.ndarray, point2: Optional[np.ndarray] = None):
        return self._distance_func(point1, point2)


class ClustersDataSpace2d(ClustersDataSpace):
    def __init__(self, x_init: np.ndarray, y_init: np.ndarray, metrics='euclidean'):
        ClustersDataSpace.__init__(self, x_init, y_init, metrics=metrics)
        self.data_ration = np.array([self.x_init,
                                     self.y_init]).transpose()


class ClustersDataSpaceFeatures(ClustersDataSpace, ABC):

    def __init__(self, x_init: np.ndarray, y_init: np.ndarray, features_init: np.ndarray, metrics='euclidean'):
        """
        :param x_init:
        :param y_init:
        :param features_init: each row corresponds to a single data point.
        """
        super().__init__(x_init, y_init, metrics=metrics)
        self.features_init = features_init.copy()
        self.data_ration = np.concatenate((ClustersData.array_rationing(self.x_init),
                                           ClustersData.array_rationing(self.y_init),
                                           ClustersData.array_rationing(self.features_init)), axis=1)
