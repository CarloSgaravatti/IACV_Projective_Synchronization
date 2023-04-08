import numpy as np
from synchronization import utils


def projective_synch(Z: np.array, A: np.array, remove_imag=False, root=None):
    iD = np.diag(1 / A.sum(axis=1))
    w, v = np.linalg.eig(np.kron(iD, np.eye(4)) @ Z)
    Q = v[:, np.argsort(w)[-4:]]  # top-4 eigenvectors
    if remove_imag:
        Q = np.real(Q)
    if root is None:
        root = np.argmax(A.sum(axis=1))  # choose the maximum degree node as the root
    return utils.scale_matrices(Q, 4, root), root
