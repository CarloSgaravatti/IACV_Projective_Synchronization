import numpy as np
from synchronization import utils
from synchronization.spanning_tree_synchronization import spanning_tree_sync


def multi_source_propagation(Z: np.ndarray, A: np.ndarray):
    """
    Computes the solution to the projective synchronization problem using the Multi-Source Propagation
    method, that averages the solution of the spanning tree approach from multiple reference nodes
    :param Z: the 4n x 4n matrix containing the pairwise projective transformations between the nodes
    :param A: the adjacency matrix of the graph
    :return: the solution to the synchronization problem, contained in a 4n x d matrix,
        and the maximum degree node used as reference node
    """
    node_degrees = A.sum(axis=1)
    n = A.shape[0]
    d = int(Z.shape[0] / n)
    # select the 10 nodes with the highest degree
    roots = np.argsort(node_degrees)[-10:]
    max_degree_node = roots[-1]
    solutions = np.empty([0, n * d, d])
    for root in roots:
        # scale the spanning tree solution w.r.t the node with the highest degree
        sol = utils.scale_matrices(spanning_tree_sync(Z, A, root), d, max_degree_node)
        # utils.normalize divide each matrix by the sum of the elements first and then by the norm
        solutions = np.concatenate([solutions, [utils.normalize_matrices(sol)]])
    return solutions.sum(axis=0) / 10, max_degree_node
