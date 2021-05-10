import pytest
import numpy as np
import sklearn.datasets
import json
import os.path

from dataclasses import dataclass, field
from typing import List
from clustering_algorithms import K_MXT
from clusters_data import ClustersDataSpace2d


@dataclass
class DataForAlgorithms:
    name: str
    x_init: np.ndarray
    y_init: np.ndarray
    k: int
    eps: float
    expected_started_graph: List[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.expected_started_graph = self.read_json_file(self.get_file_name('started_graph'))

    def read_json_file(self, file_name):
        with open(file_name) as file:
            return json.loads(file.read())

    def get_file_name(self, file_name_suffix):
        return os.path.join('./resources/', f'{self.name}_{file_name_suffix}.json')


blobs_coord = sklearn.datasets.make_blobs(n_samples=50, random_state=0, cluster_std=0.5)[0]
circles_coord = sklearn.datasets.make_circles(n_samples=50, noise=0.05, random_state=0, factor=0.4)[0]
moons_coord = sklearn.datasets.make_moons(n_samples=50, noise=0.05, random_state=0)[0]


blob_data = DataForAlgorithms(
    name='blob',
    x_init=blobs_coord[:, 0],
    y_init=blobs_coord[:, 1],
    k=9,
    eps=0.8,
)
print(blob_data)


circle_data = DataForAlgorithms(
    name='circle',
    x_init=circles_coord[:, 0],
    y_init=circles_coord[:, 1],
    k=9,
    eps=0.8,
)
print(circle_data)

moon_data = DataForAlgorithms(
    name='moon',
    x_init=moons_coord[:, 0],
    y_init=moons_coord[:, 1],
    k=9,
    eps=0.8,
)
print(moon_data)


class TestK_MXT:
    @pytest.mark.parametrize('x_init, y_init, k, eps, expected', [
        (blob_data.x_init, blob_data.y_init, blob_data.k, blob_data.eps, blob_data.expected_started_graph),
        (circle_data.x_init, circle_data.y_init, circle_data.k, circle_data.eps, circle_data.expected_started_graph),
        (moon_data.x_init, moon_data.y_init, moon_data.k, moon_data.eps, moon_data.expected_started_graph),
    ])
    def test_make_start_graph(self, x_init, y_init, k, eps, expected):
        clusters = ClustersDataSpace2d(x_init=x_init, y_init=y_init, metrics='euclidean')
        alg = K_MXT(k=k, eps=eps, clusters_data=clusters)
        alg.make_start_graph()
        # with open('./resources/moon_started_graph.json', 'w') as file:
        #     json.dump([row.tolist() for row in alg.start_graph], file)
        assert len(alg.start_graph) == len(expected)
        for i in range(len(expected)):
            curr_expected_set = set(expected[i])
            assert len(curr_expected_set) == len(alg.start_graph[i]) == len(expected[i])
            diff_elements = set(alg.start_graph[i]) ^ set(expected[i])
            assert not diff_elements


