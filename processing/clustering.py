# processing/clustering.py
from typing import List, Set
import numpy as np


class AvgLinkClustering:
    """
    Very simple average-link style clustering using a distance matrix.

    graph[i, j] in [0, 1] where 0 = best (closest), 1 = worst (farthest).
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        id_translation: List[int],
        threshold: float = 0.5,
    ) -> None:
        self.graph = dist_matrix
        self.n = dist_matrix.shape[0]
        self.visited = np.zeros(self.n, dtype=bool)
        self.threshold = threshold
        self.sort = np.argsort(self.graph, axis=None)
        self.id_translation = id_translation

    def grow_cluster(self, start_node: int) -> List[int]:
        cluster = [start_node]
        self.visited[start_node] = True

        changed = True
        while changed:
            changed = False
            new_node = -1
            dist = np.inf
            for node in range(self.n):
                if not self.visited[node]:
                    # here we use min distance to any member of the cluster
                    avg_dist = np.min(self.graph[node, cluster])
                    if dist > avg_dist <= self.threshold:
                        new_node = node
                        dist = avg_dist

            if new_node != -1:
                self.visited[new_node] = True
                changed = True
                cluster.append(new_node)
        return cluster

    def translate_cluster(self, cluster: List[int]) -> Set[int]:
        return set(self.id_translation[i] for i in cluster)

    def find_clusters(self) -> List[Set[int]]:
        clusters: List[Set[int]] = []
        for idx in self.sort:
            ob1, ob2 = np.unravel_index(idx, self.graph.shape)
            if ob1 >= ob2:
                continue
            if not self.visited[ob1]:
                cluster = self.grow_cluster(ob1)
                clusters.append(self.translate_cluster(cluster))
        return clusters
