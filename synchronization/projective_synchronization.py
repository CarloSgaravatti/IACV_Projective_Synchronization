import numpy as np


def projective_synch(Z: np.array, A: np.array, remove_imag=False, root=None):
    n = A.shape[0]
    iD = np.diag(1 / A.sum(axis=1))
    w, v = np.linalg.eig(np.kron(iD, np.eye(4)) @ Z)
    Q = v[:, np.argsort(w)[-4:]]  # top-4 eigenvectors
    if remove_imag:
        Q = np.real(Q)
    if root is None:
        root = np.argmax(A.sum(axis=1))  # choose the maximum degree node as the root
    Q_first_inv = np.linalg.inv(Q[4*root: 4*(root + 1), :])
    for i in range(n):
        Q[4*i:4*(i+1), :] = Q[4*i:4*(i+1), :] @ Q_first_inv
    return Q, root
