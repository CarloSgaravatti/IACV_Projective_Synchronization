from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_tree
import numpy as np


def spanning_tree_sync(Z: np.ndarray, A: np.ndarray):
    n = A.shape[0]
    d = int(Z.shape[0] / n)
    sparse_A = csr_matrix(A)
    T = breadth_first_tree(sparse_A, 0, directed=False)
    T_matrix = T.toarray().astype(int)
    X = [np.eye(d)] + [None] * (n -1)

    def visit_node(j):
        for k in range(n):
            if T_matrix[j, k] != 0:
                X[k] = Z[d * k:d * (k + 1), d * j:d * (j + 1)] @ X[j]
                visit_node(k)

    visit_node(0)
    X_ = np.empty([0, d])
    for i in range(n):
        X_ = np.concatenate([X_, X[i]], axis=0)
    return X_, 0
