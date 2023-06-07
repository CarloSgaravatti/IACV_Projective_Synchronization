import numpy as np
from synchronization import utils


def projective_synch(Z: np.array, A: np.array, remove_imag=False, root=None):
    """
    Solves the projective synchronization problem using the spectral solution (extended to complex
    numbers) approach
    :param Z: the 4n x 4n matrix containing the pairwise projective transformations between the nodes
    :param A: the adjacency matrix of the graph
    :param remove_imag: if true the imaginary values are removed before scaling, otherwise a complex solution
        is returned by the function
    :param root: the reference node for the final scaling, if this is None, the node with the highest degree will
        be computed and used as a reference
    :return: the solution to the synchronization problem, contained in a 4n x d matrix, together with the used reference node
    """
    iD = np.diag(1 / A.sum(axis=1))
    w, v = np.linalg.eig(np.kron(iD, np.eye(4)) @ Z)
    Q = v[:, np.argsort(w)[-4:]]  # top-4 eigenvectors
    if remove_imag:
        Q = np.real(Q)
    if root is None:
        root = np.argmax(A.sum(axis=1))  # choose the maximum degree node as the root
    return utils.scale_matrices(Q, 4, root), root
