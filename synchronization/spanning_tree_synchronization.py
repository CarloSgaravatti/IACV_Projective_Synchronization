from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_tree
import numpy as np


def spanning_tree_sync(Z: np.ndarray, A: np.ndarray, root: int):
    """
    Computes the solution to the general synchronization problem in GL(d) (homography synchronization for
    d = 3, projective synchronization for d = 4) using a spanning tree approach
    :param Z: the dn x dn matrix containing the pairwise measurements between the nodes
    :param A: the adjacency matrix of the graph
    :param root: the nodes that will be used as the root of the spanning tree
    :return: the solution to the synchronization problem, contained in a dn x d matrix
    """
    n = A.shape[0]
    d = int(Z.shape[0] / n)
    sparse_A = csr_matrix(A)
    T = breadth_first_tree(sparse_A, root, directed=False)
    T_matrix = T.toarray().astype(int)
    X = [np.eye(d)] + [None] * (n - 1)
    X = [None] * root + [np.eye(d)] + [None] * (n - root - 1)

    def visit_node(j):
        for k in range(n):
            if T_matrix[j, k] != 0:
                X[k] = Z[d * k:d * (k + 1), d * j:d * (j + 1)] @ X[j]
                visit_node(k)

    visit_node(root)
    X_ = np.empty([0, d])
    for i in range(n):
        X_ = np.concatenate([X_, X[i]], axis=0)
    return X_
