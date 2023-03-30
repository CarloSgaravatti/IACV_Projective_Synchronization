import numpy as np


def homography_synch(Z: np.array, A: np.array) -> (np.ndarray, int):
    n = A.shape[0]

    iD = np.diag(1 / A.sum(axis=1))
    w, v = np.linalg.eig(np.kron(iD, np.eye(3)) @ Z)
    Q = v[:, np.argsort(w)[-3:]]  # top-3 eigenvectors
    root = np.argmax(A.sum(axis=1))
    Q_first_inv = np.linalg.inv(Q[3*root: 3*(root + 1), :])
    for i in range(n):
        Q[3*i:3*(i+1), :] = Q[3*i:3*(i+1), :] @ Q_first_inv
    Q = np.real(Q)

    H = np.empty([0, 3])
    for i in range(n):
        det_qi = np.linalg.det(Q[3*i:3*(i+1), :])
        Q_i = Q[3*i:3*(i+1), :] / (np.sign(det_qi) * np.power(np.abs(det_qi), 1/3))
        H = np.concatenate([H, Q_i], axis=0)
    return H, root



