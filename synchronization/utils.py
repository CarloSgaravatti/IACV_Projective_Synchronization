import numpy as np


def scale_matrices(X: np.ndarray, d: int, ref: int):
    ref_matrix_inv = np.linalg.inv(X[d*ref: d*(ref + 1), :])
    X_scaled = np.zeros(X.shape)
    n = int(X.shape[0] / d)
    for i in range(n):
        X_scaled[d*i: d*(i + 1), :] = X[d*i: d*(i + 1), :] @ ref_matrix_inv
    return X_scaled


def calculate_matrix_angle(X1: np.ndarray, X2: np.ndarray, d: int):
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
    return np.linalg.norm(X1 - X2)


def get_error(U: np.ndarray, X: np.ndarray, d: int, distance_type='angle'):
    n = int(U.shape[0] / d)
    err = 0
    distance_func = calculate_matrix_angle if distance_type == 'angle' else calculate_matrix_distance
    for i in range(n):
        err += distance_func(U[d*i: d*(i+1), :], X[d*i: d*(i+1), :], d)
    return err / n


