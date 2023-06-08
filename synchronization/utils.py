import numpy as np


def scale_matrices(X: np.ndarray, d: int, ref: int):
    """
    Scale each matrices of the input block matrix by multiplying each of them by the inverse
    of the matrix X[d*ref: d*(ref + 1), :]
    :param X: the nd x d block matrix to be scaled, where the i-th block is the matrix X[d*i: d*(i + 1), :]
    :param d: the dimension of matrices
    :param ref: the index of the reference matrix
    :return: the scaled block matrix
    """

    ref_matrix_inv = np.linalg.inv(X[d*ref: d*(ref + 1), :])
    X_scaled = np.zeros(X.shape, dtype=X.dtype)
    n = int(X.shape[0] / d)
    for i in range(n):
        X_scaled[d*i: d*(i + 1), :] = X[d*i: d*(i + 1), :] @ ref_matrix_inv
    return X_scaled


def normalize_matrices(X: np.ndarray):
    """
    Normalize each matrix of the input nd x d block matrix by dividing each block matrix w.r.t the sum of the elements
    and, then, w.r.t. the norm of the vectorized form of the matrix
    :param X: the block matrix to be normalized
    :return: the normalized block matrix
    """

    X_normalized = np.zeros(X.shape, dtype=X.dtype)
    d = X.shape[1]
    n = int(X.shape[0] / d)
    for i in range(n):
        x_vec = X[d*i: d*(i + 1), :].reshape((d ** 2, 1))
        x_vec_sum = x_vec.sum()
        X_normalized[d*i: d*(i + 1), :] = X[d*i: d*(i + 1), :] / x_vec_sum
        norm = np.linalg.norm(x_vec / x_vec_sum)
        X_normalized[d * i: d * (i + 1), :] = X_normalized[d * i: d * (i + 1), :] / norm
    return X_normalized


def calculate_matrix_angle(X1: np.ndarray, X2: np.ndarray, d: int):
    """
    Calculates the angle between the vectorized form of the two input matrices
    :param X1: a d x d matrix
    :param X2: a d x d matrix
    :param d: the dimension of the matrices
    :return: the angle between the two matrices
    """

    x1 = X1.reshape((d ** 2, 1))
    x2 = X2.reshape((d ** 2, 1))
    x1 = np.real(x1)
    x2 = np.real(x2)
    cos_alpha_1 = (x1.T @ x2)[0, 0] / (np.linalg.norm(x2) * np.linalg.norm(x1))
    cos_alpha_1 = 1 if np.abs(cos_alpha_1) > 1.0 else cos_alpha_1
    alpha_rad_1 = np.abs(np.arccos(cos_alpha_1))
    cos_alpha_2 = (x1.T @ (-x2))[0, 0] / (np.linalg.norm(x2) * np.linalg.norm(x1))
    cos_alpha_2 = 1 if np.abs(cos_alpha_2) > 1.0 else cos_alpha_2
    alpha_rad_2 = np.abs(np.arccos(cos_alpha_2))
    alpha_rad = alpha_rad_1 if alpha_rad_1 <= alpha_rad_2 else alpha_rad_2
    return alpha_rad * 180 / np.pi


def calculate_matrix_distance(X1: np.ndarray, X2: np.ndarray, d: int):
    """
    Calculates the distance in terms of norm between the two input matrices
    :param X1: a d x d matrix
    :param X2: a d x d matrix
    :param d: the dimension of the matrices
    :return: the distance between the two matrices
    """

    return np.linalg.norm(X1 - X2)


def get_error(U: np.ndarray, X: np.ndarray, d: int, distance_type='angle'):
    """
    Calculates the distance, in terms of norm or angle, between the two input nd x d block matrices
    :param U: a nd x d block matrix
    :param X: a nd x d block matrix
    :param d: the dimension of the blocks of the matrix
    :param distance_type: the type of distance that will be used (angle or norm)
    :return: the mean distance between the blocks of the two matrices
    """

    n = int(U.shape[0] / d)
    err = 0
    distance_func = calculate_matrix_angle if distance_type == 'angle' else calculate_matrix_distance
    for i in range(n):
        err += distance_func(U[d*i: d*(i+1), :], X[d*i: d*(i+1), :], d)
    return err / n


