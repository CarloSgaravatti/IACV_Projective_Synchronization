import numpy as np
from synchronization.projective_synchronization_spectral import projective_synch
from homography_synchronization import delete_info
import synchronization.utils as utils
import matplotlib.pyplot as plt


def build_z_projective_tr(n: int, sigma: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X = [np.empty([0, 4], dtype=complex)] * 4
    X_not_scaled = np.empty([0, 4])
    X_b = [np.empty([4, 0], dtype=complex)] * 4
    for _ in range(n):
        X_i = np.random.randn(4, 4)
        X_not_scaled = np.concatenate([X_not_scaled, X_i], axis=0)
        det_xi = np.linalg.det(X_i)
        det_xi_roots = np.power(det_xi, 1 / 4, dtype=complex) * np.exp(2j * np.pi * np.arange(4) / 4)
        for i in range(4):
            X[i] = np.concatenate([X[i], X_i / det_xi_roots[i]], axis=0)
            X_b[i] = np.concatenate([X_b[i], np.linalg.inv(X_i / det_xi_roots[i])], axis=1)
    Z = [None] * 4
    for i in range(4):
        Z[i] = X[i] @ X_b[i] + np.random.randn(n * 4, n * 4) * sigma
    A = np.ones((n, n))
    return Z, A, X_not_scaled


def test(n: int, sigma: float, miss_rate: float):
    Z, A, X = build_z_projective_tr(n, sigma)
    A = delete_info(A, int(miss_rate * np.sum(np.arange(n - 1))), n)
    U_root_45, root = projective_synch(Z[0], A)
    U_root_135, _ = projective_synch(Z[1], A)
    U_root_225, _ = projective_synch(Z[2], A)
    U_root_315, _ = projective_synch(Z[3], A)
    X_scaled = utils.scale_matrices(X, 4, root)
    err_45 = utils.get_error(U_root_45, X_scaled, 4, distance_type='angle')
    err_135 = utils.get_error(U_root_135, X_scaled, 4, distance_type='angle')
    err_225 = utils.get_error(U_root_225, X_scaled, 4, distance_type='angle')
    err_315 = utils.get_error(U_root_315, X_scaled, 4, distance_type='angle')
    return err_45, err_135, err_225, err_315


def test_different_noise(n: int, miss_rate: float, num_repeat: int):
    sigmas = np.concatenate([[0], np.logspace(0, 6, 7) * 1e-6], axis=0)
    errors_45, errors_135, errors_225, errors_315 = list(), list(), list(), list()
    for sigma in sigmas:
        print(sigma)
        err_45, err_135, err_225, err_315 = 0, 0, 0, 0
        for _ in range(num_repeat):
            res = test(n, miss_rate=miss_rate, sigma=sigma)
            err_45, err_135, err_225, err_315 = err_45 + res[0], err_135 + res[1], err_225 + res[2], err_315 + res[3]
        err_45, err_135, err_225, err_315 = err_45 / num_repeat, err_135 / num_repeat, err_225 / num_repeat, err_315 / num_repeat
        errors_45.append(err_45)
        errors_135.append(err_135)
        errors_225.append(err_225)
        errors_315.append(err_315)
    plt.figure()
    plt.plot(sigmas, errors_45, 'o-', label=f'root 45')
    plt.plot(sigmas, errors_135, 'o-', label=f'root 135')
    plt.plot(sigmas, errors_225, 'o-', label=f'root 225')
    plt.plot(sigmas, errors_315, 'o-', label=f'root 315')
    plt.yscale('log')
    plt.xscale('symlog', linthresh=1e-6)
    plt.xlabel('noise')
    plt.ylabel('error')
    plt.xlim(left=-1e-7)
    plt.title(f'Spectral solution: different roots comparison (%{int(miss_rate * 100)} missing edges, {n} nodes)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_different_noise(100, 0.8, 20)
