import numpy as np
from synchronization import utils


def homography_synch(Z: np.array, A: np.array) -> (np.ndarray, int):
    """
    Performs homography synchronization on the given data
    :param Z: the 3n x 3n matrix containing the pairwise measurements
    :param A: the adjacency matrix of the graph
    :return: the solution of the problem, contained in a 3n x d matrix,
        and the reference node (the node with the highest degree)
    """
    n = A.shape[0]

    iD = np.diag(1 / A.sum(axis=1))  # inverse of the degree matrix
    w, v = np.linalg.eig(np.kron(iD, np.eye(3)) @ Z)
    Q = v[:, np.argsort(w)[-3:]]  # top-3 eigenvectors
    root = np.argmax(A.sum(axis=1))  # the reference node is the one with the maximum degree
    Q = np.real(utils.scale_matrices(Q, 3, root))

    H = np.empty([0, 3])
    for i in range(n):
        # make the determinant of the matrices equal to one
        det_qi = np.linalg.det(Q[3*i:3*(i+1), :])
        Q_i = Q[3*i:3*(i+1), :] / (np.sign(det_qi) * np.power(np.abs(det_qi), 1/3))
        H = np.concatenate([H, Q_i], axis=0)
    return H, root



