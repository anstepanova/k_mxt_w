import pytest
import numpy as np

from unittest.mock import patch
from k_mxt_w.clusters_data import *


class TestClustersData:
    @pytest.mark.parametrize('array, rationed_array', [
        (np.array([1, 2, 3, 4]), np.array([-1.34164079, -0.4472136, 0.4472136, 1.34164079])),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])),
        (np.array([10, 10]), np.array([10, 10])),
        (np.array([10, -10]), np.array([1, -1])),
        (np.array([0, 0.5, 0.33, 0.9]), np.array([-1.33365272, 0.20814233, -0.31606799, 1.44157838])),
    ])
    def test_array_rationing(self, array, rationed_array):
        np.testing.assert_allclose(rationed_array, ClustersData.array_rationing(array))

    @pytest.mark.parametrize('graph, cluster_numbers, expected', [
        ([
            [1, 2],
            [0, 2, 3],
            [0, 1],
            [1, 6, 7],
            [5, 6],
            [4, 6],
            [3, 4, 5],
            [3, 8, 9],
            [7, 9],
            [7, 8]
         ], [0, 0, 0, 0, 1, 1, 1, 2, 2, 2], 0.4895833333333332),
        ([
            [0],
            [1]
        ], [0, 1], 0.5),
    ])
    def test_calculate_modularity(self, graph, cluster_numbers, expected):
        p = patch.multiple(ClustersData, __abstractmethods__=set())
        p.start()
        clusters = ClustersData()
        clusters.cluster_numbers = cluster_numbers
        clusters.num_of_data = len(clusters.cluster_numbers)
        modularity_value = clusters.calculate_modularity(graph)
        p.stop()
        np.testing.assert_almost_equal(modularity_value, expected)


class TestMetrics:
    @pytest.mark.parametrize('data, point1, point2, result', [
        (np.array([[0, 0], [1, 1]]), 0, 1, 1.4142135623730951),
        (np.array([[0, 0], [2, 2]]), 0, 1, 2.8284271247461903),
        (np.array([[0, 0], [4, 4]]), 0, 1, 5.656854249492381),
        (np.array([[0, 0, 0], [4, 4, 1]]), 0, 1, 5.744562646538029),
        (np.array([[-1, -2], [2, 2]]), 0, 1, 5.0),
        (np.array([[-1, -2], [-1, -2]]), 0, 1, 0.0),
        (np.array([[-1, -2], [1, -2]]), 0, 1, 2.0),
        (np.array([[-1, -2], [-1, 2]]), 0, 1, 4.0),
        (np.array([[-1, -2], [1, 2]]), 0, 1, 4.47213595499958),
        (np.array([[-1, -2], [-1, -2]]), 0, 1, 0.0),
        (np.array([[1, -2], [-1, -2]]), 0, 1, 2.0),
        (np.array([[-1, 2], [-1, -2]]), 0, 1, 4.0),
        (np.array([[1, 2], [-1, -2]]), 0, 1, 4.47213595499958),
        (np.array([[0, 0, 0], [4, 4, 1]]), 0, 1, 5.744562646538029),
        (np.array([[0, 0], [2, 2]]), 0, None, [0, 2.82842712]),
        (np.array([[0, 0], [4, 4]]), 0, None, [0, 5.656854249492381]),
        (np.array([[0, 0, 0], [4, 4, 1]]), 1, None, [5.744562646538029, 0]),
        (np.array([[-1, -2], [2, 2]]), 0, None, [0, 5.0]),
        (np.array([[-1, -2], [-1, -2]]), 0, None, [0.0, 0.0]),
        (np.array([[-1, -2], [1, -2]]), 0, None, [0, 2.0]),
        (np.array([[-1, -2], [-1, 2]]), 1, None, [4.0, 0]),
        (np.array([[-1, -2], [1, 2]]), 0, None, [0, 4.47213595499958]),
    ])
    def test_euclidean_distance(self, data, point1, point2, result):
        metrics = MetricsMixin()
        metrics.data_ration = data
        np.testing.assert_almost_equal(metrics.euclidean_distance(point1, point2), result)
        np.testing.assert_allclose(metrics.data_ration, data)

    @pytest.mark.parametrize('data, point1, point2, result', [
        (np.array([[0, 0], [1, 1]]), 0, 1, 2),
        (np.array([[0, 0], [2, 2]]), 0, 1, 4),
        (np.array([[0, 0], [4, 4]]), 0, 1, 8),
        (np.array([[0, 0, 0], [4, 4, 1]]), 0, 1, 9),
        (np.array([[-1, -2], [2, 2]]), 0, 1, 7),
        (np.array([[-1, -2], [-1, -2]]), 0, 1, 0.0),
        (np.array([[-1, -2], [1, -2]]), 0, 1, 2),
        (np.array([[-1, -2], [-1, 2]]), 0, 1, 4.0),
        (np.array([[-1, -2], [1, 2]]), 0, 1, 6),
        (np.array([[-1, -2], [-1, -2]]), 0, 1, 0.0),
        (np.array([[1, -2], [-1, -2]]), 0, 1, 2.0),
        (np.array([[-1, 2], [-1, -2]]), 0, 1, 4.0),
        (np.array([[1, 2], [-1, -2]]), 0, 1, 6),
        (np.array([[0, 0, 0], [4, 4, 1]]), 0, 1, 9),
        (np.array([[0, 0], [2, 2]]), 0, None, [0, 4.]),
        (np.array([[0, 0], [4, 4]]), 0, None, [0, 8.]),
        (np.array([[0, 0, 0], [4, 4, 1]]), 1, None, [9., 0]),
        (np.array([[-1, -2], [2, 2]]), 0, None, [0, 7.]),
        (np.array([[-1, -2], [-1, -2]]), 0, None, [0.0, 0.0]),
        (np.array([[-1, -2], [1, -2]]), 0, None, [0, 2.0]),
        (np.array([[-1, -2], [-1, 2]]), 1, None, [4.0, 0]),
        (np.array([[-1, -2], [1, 2]]), 0, None, [0, 6.]),
    ])
    def test_manhattan_distance(self, data, point1, point2, result):
        metrics = MetricsMixin()
        metrics.data_ration = data
        np.testing.assert_almost_equal(metrics.manhattan_distance(point1, point2), result)
        np.testing.assert_allclose(metrics.data_ration, data)

