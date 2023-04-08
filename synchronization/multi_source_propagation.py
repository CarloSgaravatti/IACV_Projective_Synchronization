import numpy as np
from synchronization import utils
from synchronization.spanning_tree_synchronization import spanning_tree_sync


def multi_source_propagation(Z: np.ndarray, A: np.ndarray):
    node_degrees = A.sum(axis=1)
    n = A.shape[0]
    d = int(Z.shape[0] / n)
    # select the 10 nodes with the highest degree
    roots = np.argsort(node_degrees)[-10:]
    max_degree_node = roots[-1]
    solutions = np.empty([0, n * d, d])
    for root in roots:
        sol = utils.scale_matrices(spanning_tree_sync(Z, A, root), d, max_degree_node)
        solutions = np.concatenate([solutions, [utils.normalize_matrices(sol)]])
    return solutions.sum(axis=0) / 10, max_degree_node, solutions[-2], solutions[-1]
