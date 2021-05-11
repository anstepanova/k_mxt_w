import pytest
import numpy as np
import sklearn.datasets
import json
import os.path

from dataclasses import dataclass, field
from typing import List, Union

import clustering_algorithms
from clustering_algorithms import K_MXT
from clusters_data import ClustersDataSpace2d


@dataclass
class DataForAlgorithms:
    name: str
    cls: Union[clustering_algorithms.K_MXT, clustering_algorithms.K_MXT_gauss]
    x_init: np.ndarray
    y_init: np.ndarray
    k: int
    eps: float
    expected_started_graph: List[np.ndarray] = field(init=False)
    expected_get_arc_weight: dict = field(init=False)
    expected_k_graph: List[np.ndarray] = field(init=False)
    expected_clustering_result: np.ndarray = field(init=False)

    def __post_init__(self):
        self.expected_started_graph = self.read_json_file(self.get_file_name('started_graph'))
        self.expected_get_arc_weight = self.read_json_file(self.get_file_name('get_arc_weight'))
        self.expected_k_graph = self.read_json_file(self.get_file_name('k_graph'))
        self.expected_clustering_result = np.array(self.read_json_file(self.get_file_name('clustering_result')))

    def read_json_file(self, file_name):
        with open(file_name) as file:
            return json.loads(file.read())

    def get_file_name(self, file_name_suffix):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, f'resources/{self.cls.__name__}', f'{self.name}_{file_name_suffix}.json')


blobs_coord = sklearn.datasets.make_blobs(n_samples=50, random_state=0, cluster_std=0.5)[0]
circles_coord = sklearn.datasets.make_circles(n_samples=50, noise=0.05, random_state=0, factor=0.4)[0]
moons_coord = sklearn.datasets.make_moons(n_samples=50, noise=0.05, random_state=0)[0]


blob_data = DataForAlgorithms(
    name='blob',
    cls=clustering_algorithms.K_MXT,
    x_init=blobs_coord[:, 0],
    y_init=blobs_coord[:, 1],
    k=9,
    eps=0.8,
)


circle_data = DataForAlgorithms(
    name='circle',
    cls=clustering_algorithms.K_MXT,
    x_init=circles_coord[:, 0],
    y_init=circles_coord[:, 1],
    k=9,
    eps=0.8,
)

moon_data = DataForAlgorithms(
    name='moon',
    cls=clustering_algorithms.K_MXT,
    x_init=moons_coord[:, 0],
    y_init=moons_coord[:, 1],
    k=9,
    eps=0.8,
)

# blob_data = DataForAlgorithms(
#     name='blob',
#     cls=clustering_algorithms.K_MXT,
#     x_init=blobs_coord[:, 0],
#     y_init=blobs_coord[:, 1],
#     k=9,
#     eps=0.8,
# )
#
#
# circle_data = DataForAlgorithms(
#     name='circle',
#     cls=clustering_algorithms.K_MXT,
#     x_init=circles_coord[:, 0],
#     y_init=circles_coord[:, 1],
#     k=9,
#     eps=0.8,
# )
#
# moon_data = DataForAlgorithms(
#     name='moon',
#     cls=clustering_algorithms.K_MXT,
#     x_init=moons_coord[:, 0],
#     y_init=moons_coord[:, 1],
#     k=9,
#     eps=0.8,
# )


class TestK_MXT:
    @pytest.mark.parametrize('cls, x_init, y_init, k, eps', [
        (blob_data.cls, blob_data.x_init, blob_data.y_init, blob_data.k, blob_data.eps),
        (circle_data.cls, circle_data.x_init, circle_data.y_init, circle_data.k, circle_data.eps),
        (moon_data.cls, moon_data.x_init, moon_data.y_init, moon_data.k, moon_data.eps),
    ])
    def test_init(self, cls, x_init, y_init, k, eps):
        clusters = ClustersDataSpace2d(x_init=x_init, y_init=y_init, metrics='euclidean')
        alg = cls(k=k, eps=eps, clusters_data=clusters)
        assert alg.k == k
        np.testing.assert_almost_equal(alg.eps, eps)
        assert alg.clusters_data is clusters
        assert alg.num_of_vertices == clusters.num_of_data == len(x_init)
        assert len(alg.start_graph) == alg.num_of_vertices
        assert set(alg.start_graph) == {None}
        assert len(alg.k_graph) == alg.num_of_vertices
        assert set(alg.k_graph) == {None}


    @pytest.mark.parametrize('cls, x_init, y_init, k, eps, expected', [
        (blob_data.cls, blob_data.x_init, blob_data.y_init, blob_data.k, blob_data.eps, blob_data.expected_started_graph),
        (circle_data.cls, circle_data.x_init, circle_data.y_init, circle_data.k, circle_data.eps, circle_data.expected_started_graph),
        (moon_data.cls, moon_data.x_init, moon_data.y_init, moon_data.k, moon_data.eps, moon_data.expected_started_graph),
    ])
    def test_make_start_graph(self, cls, x_init, y_init, k, eps, expected):
        clusters = ClustersDataSpace2d(x_init=x_init, y_init=y_init, metrics='euclidean')
        alg = cls(k=k, eps=eps, clusters_data=clusters)
        alg.make_start_graph()
        # with open('./resources/moon_started_graph.json', 'w') as file:
        #     json.dump([row.tolist() for row in alg.start_graph], file)
        assert len(alg.start_graph) == len(expected)
        for i in range(len(expected)):
            curr_expected_set = set(expected[i])
            assert len(curr_expected_set) == len(alg.start_graph[i]) == len(expected[i])
            diff_elements = set(alg.start_graph[i]) ^ set(expected[i])
            assert not diff_elements

    @pytest.mark.parametrize('cls, x_init, y_init, k, eps, expected',[
        (blob_data.cls, blob_data.x_init, blob_data.y_init, blob_data.k, blob_data.eps, blob_data.expected_get_arc_weight),
        (circle_data.cls, circle_data.x_init, circle_data.y_init, circle_data.k, circle_data.eps, circle_data.expected_get_arc_weight),
        (moon_data.cls, moon_data.x_init, moon_data.y_init, moon_data.k, moon_data.eps, moon_data.expected_get_arc_weight),
    ])
    def test_get_arc_weight(self, cls, x_init, y_init, k, eps, expected):
        def get_arc_name(v1, v2):
            return f'({v1}, {v2})'

        clusters = ClustersDataSpace2d(x_init=x_init, y_init=y_init, metrics='euclidean')
        alg = cls(k=k, eps=eps, clusters_data=clusters)
        alg.make_start_graph()
        for v in range(alg.num_of_vertices):
            for to in range(alg.num_of_vertices):
                assert alg.get_arc_weight(v, to) == expected[get_arc_name(v, to)]
                assert alg.get_arc_weight(v, to) == expected[get_arc_name(to, v)]
        # with open('./resources/moon_get_arc_weight.json', 'w') as file:
        #     json.dump(weights, file)
        # raise NotImplementedError

    @pytest.mark.parametrize('cls, x_init, y_init, k, eps, expected', [
        (blob_data.cls, blob_data.x_init, blob_data.y_init, blob_data.k, blob_data.eps, blob_data.expected_k_graph),
        (circle_data.cls, circle_data.x_init, circle_data.y_init, circle_data.k, circle_data.eps, circle_data.expected_k_graph),
        (moon_data.cls, moon_data.x_init, moon_data.y_init, moon_data.k, moon_data.eps, moon_data.expected_k_graph),
    ])
    def test_make_k_graph(self, cls, x_init, y_init, k, eps, expected):
        clusters = ClustersDataSpace2d(x_init=x_init, y_init=y_init, metrics='euclidean')
        alg = cls(k=k, eps=eps, clusters_data=clusters)
        alg.make_start_graph()
        alg.make_k_graph()
        assert len(alg.k_graph) == len(expected)
        for i in range(len(expected)):
            curr_expected_set = set(expected[i])
            assert len(curr_expected_set) == len(alg.k_graph[i]) == len(expected[i])
            diff_elements = set(alg.k_graph[i]) ^ set(expected[i])
            assert not diff_elements
        # with open('./resources/moon_k_graph.json', 'w') as file:
        #     json.dump([row.tolist() for row in alg.k_graph], file)

    @pytest.mark.parametrize('cls, x_init, y_init, k, eps, expected', [
        (blob_data.cls, blob_data.x_init, blob_data.y_init, blob_data.k, blob_data.eps, blob_data.expected_clustering_result),
        (circle_data.cls, circle_data.x_init, circle_data.y_init, circle_data.k, circle_data.eps, circle_data.expected_clustering_result),
        (moon_data.cls, moon_data.x_init, moon_data.y_init, moon_data.k, moon_data.eps, moon_data.expected_clustering_result),
    ])
    def test_clustering_result(self, cls, x_init, y_init, k, eps, expected):
        clusters = ClustersDataSpace2d(x_init=x_init, y_init=y_init, metrics='euclidean')
        alg = cls(k=k, eps=eps, clusters_data=clusters)
        alg()
        np.testing.assert_array_equal(alg.clusters_data.cluster_numbers, expected)
        # with open('./resources/moon_clustering_result.json', 'w') as file:
        #     json.dump(alg.clusters_data.cluster_numbers.tolist(), file)





