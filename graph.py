import copy
import numpy as np
import collections
import logging

from typing import Optional, List


logger = logging.getLogger('k_mxt_w.clustering_algorithm')


class Graph:
    """
    Class which works with graph
    """
    def __init__(self, adj: List[List[int]]):
        """

        :param adj: adjacency list of the graph
        """
        logger.info(f'init graph')
        self.adj = copy.deepcopy(adj)
        self.num_vertices = len(self.adj)

    def get_reverse_graph(self):
        """
        Builds and returns reverse graph
        :return:
        """
        logger.info(f'start reversing a graph')
        adj = [[] for _ in range(self.num_vertices)]
        for v in range(self.num_vertices):
            for to in self.adj[v]:
                adj[to].append(v)
        return Graph(adj=adj)

    def dfs(self, vertex_order: List[int], is_save_processing_order: bool):
        """
        Makes Depth First Search of the graph
        :param vertex_order: list of vertices in traversal order
        :param is_save_processing_order: True means that the processing_order will be safe
        :return:
        """
        logger.info(f'start dfs')
        num_ucc = [0 for _ in range(self.num_vertices)]
        processing_order = None
        if is_save_processing_order:
            processing_order = []

        def dfs(start_vertex):
            logger.info(f'dfs vertex-{start_vertex}')
            call_stack = collections.deque()
            logger.info(f'dfs vertex-{start_vertex}. Create call_stack')
            call_stack.append(start_vertex)
            num_ucc[start_vertex] = cnt_call
            logger.info(f'dfs vertex-{start_vertex}. Append vertex to call_stack')
            while call_stack:
                vertex = call_stack.popleft()
                logger.info(f'dfs vertex-{vertex}. Pop vertex from call_stack')
                num_ucc[vertex] = cnt_call
                for neighbor in self.adj[vertex]:
                    if not num_ucc[neighbor]:
                        # dfs(neighbor)
                        call_stack.append(neighbor)
                        num_ucc[neighbor] = cnt_call
                        logger.info(f'dfs vertex-{vertex}. Append {neighbor} to call_stack')
                if is_save_processing_order:
                    processing_order.append(vertex)
            call_stack = None

        cnt_call = 0
        for v in vertex_order:
            if not num_ucc[v]:
                cnt_call += 1
                dfs(v)
        if is_save_processing_order:
            return num_ucc, processing_order
        return num_ucc

    def top_sort(self):
        """
        Builds topological sort
        :return: order of topological sort
        """
        logger.info(f'start topological sort')
        vertex_order = [x for x in range(self.num_vertices)]
        _, order_top_sort = self.dfs(vertex_order=vertex_order, is_save_processing_order=True)
        order_top_sort.reverse()
        return order_top_sort

    def find_scc(self):
        """
        Finds strong connected component
        :return: strong connected component of the graph
        """
        g_rev = self.get_reverse_graph()
        order = g_rev.top_sort()
        g_rev = None
        return self.dfs(vertex_order=order, is_save_processing_order=False)






