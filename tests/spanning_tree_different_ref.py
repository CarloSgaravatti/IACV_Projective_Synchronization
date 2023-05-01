import numpy as np
from synchronization.spanning_tree_synchronization import spanning_tree_sync
from homography_synchronization import delete_info
from projective_spanning_spectral_comparison import build_z_projective_tr
import synchronization.utils as utils
import matplotlib.pyplot as plt


def test_aligned(n: int, sigma: float, miss_rate: float):
    Z, A, X_not_scaled, Z_not_scaled = build_z_projective_tr(n, 4)
    Z += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    Z_not_scaled += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    A = delete_info(A, int(miss_rate * np.sum(np.arange(n - 1))), n)
    node_degrees = A.sum(axis=1)
    roots = np.argsort(node_degrees)
    U_max1 = spanning_tree_sync(Z_not_scaled, A, root=roots[-1])
    U_max2 = spanning_tree_sync(Z_not_scaled, A, root=roots[-2])
    U_med = spanning_tree_sync(Z_not_scaled, A, root=roots[int(n / 2)])
    U_min = spanning_tree_sync(Z_not_scaled, A, root=roots[0])
    X_scaled = utils.scale_matrices(X_not_scaled, 4, roots[-1])
    err_max1 = utils.get_error(U_max1, X_scaled, 4, distance_type='angle')
    err_max2 = utils.get_error(utils.scale_matrices(U_max2, 4, roots[-1]), X_scaled, 4, distance_type='angle')
    err_med = utils.get_error(utils.scale_matrices(U_med, 4, roots[-1]), X_scaled, 4, distance_type='angle')
    err_min = utils.get_error(utils.scale_matrices(U_min, 4, roots[-1]), X_scaled, 4, distance_type='angle')
    return err_max1, err_max2, err_med, err_min


def test_not_aligned(n: int, sigma: float, miss_rate: float):
    Z, A, X_not_scaled, Z_not_scaled = build_z_projective_tr(n, 4)
    Z += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    Z_not_scaled += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    A = delete_info(A, int(miss_rate * np.sum(np.arange(n - 1))), n)
    node_degrees = A.sum(axis=1)
    roots = np.argsort(node_degrees)
    U_max1 = spanning_tree_sync(Z_not_scaled, A, root=roots[-1])
    U_max2 = spanning_tree_sync(Z_not_scaled, A, root=roots[-2])
    U_med = spanning_tree_sync(Z_not_scaled, A, root=roots[int(n / 2)])
    U_min = spanning_tree_sync(Z_not_scaled, A, root=roots[0])
    err_max1 = utils.get_error(U_max1, utils.scale_matrices(X_not_scaled, 4, roots[-1]), 4, distance_type='angle')
    err_max2 = utils.get_error(U_max2, utils.scale_matrices(X_not_scaled, 4, roots[-2]), 4, distance_type='angle')
    err_med = utils.get_error(U_med, utils.scale_matrices(X_not_scaled, 4, roots[int(n / 2)]), 4, distance_type='angle')
    err_min = utils.get_error(U_min, utils.scale_matrices(X_not_scaled, 4, roots[0]), 4, distance_type='angle')
    return err_max1, err_max2, err_med, err_min


def test_different_noise_aligned(n: int, miss_rate: float, num_repeat: int):
    sigmas = np.concatenate([[0], np.logspace(0, 6, 7) * 1e-6], axis=0)
    errors_max1, errors_max2, errors_med, errors_min = list(), list(), list(), list()
    for sigma in sigmas:
        err_max1, err_max2, err_med, err_min = 0, 0, 0, 0
        for _ in range(num_repeat):
            res = test_aligned(n, miss_rate=miss_rate, sigma=sigma)
            err_max1, err_max2, err_med, err_min = err_max1 + res[0], err_max2 + res[1], err_med + res[2], err_min + res[3]
        err_max1, err_max2, err_med, err_min = err_max1 / num_repeat, err_max2 / num_repeat, err_med / num_repeat, err_min / num_repeat
        errors_max1.append(err_max1)
        errors_max2.append(err_max2)
        errors_med.append(err_med)
        errors_min.append(err_min)
    plt.figure()
    plt.plot(sigmas, errors_max1, 'o-', label=f'max degree 1')
    plt.plot(sigmas, errors_max2, 'o-', label=f'max degree 2')
    plt.plot(sigmas, errors_med, 'o-', label=f'median degree')
    plt.plot(sigmas, errors_min, 'o-', label=f'min degree')
    plt.yscale('log')
    plt.xscale('symlog', linthresh=1e-6)
    plt.xlabel('noise')
    plt.ylabel('error')
    plt.xlim(left=-1e-7)
    plt.title(f'Projective synchronization with %{int(miss_rate * 100)} of missing edges, and {n} nodes')
    plt.legend()
    plt.show()


def test_different_noise_not_aligned(n: int, miss_rate: float, num_repeat: int):
    sigmas = np.concatenate([[0], np.logspace(0, 6, 7) * 1e-6], axis=0)
    errors_max1, errors_max2, errors_med, errors_min = list(), list(), list(), list()
    for sigma in sigmas:
        err_max1, err_max2, err_med, err_min = 0, 0, 0, 0
        for _ in range(num_repeat):
            res = test_not_aligned(n, miss_rate=miss_rate, sigma=sigma)
            err_max1, err_max2, err_med, err_min = err_max1 + res[0], err_max2 + res[1], err_med + res[2], err_min + res[3]
        err_max1, err_max2, err_med, err_min = err_max1 / num_repeat, err_max2 / num_repeat, err_med / num_repeat, err_min / num_repeat
        errors_max1.append(err_max1)
        errors_max2.append(err_max2)
        errors_med.append(err_med)
        errors_min.append(err_min)
    plt.figure()
    plt.plot(sigmas, errors_max1, 'o-', label=f'max degree 1')
    plt.plot(sigmas, errors_max2, 'o-', label=f'max degree 2')
    plt.plot(sigmas, errors_med, 'o-', label=f'median degree')
    plt.plot(sigmas, errors_min, 'o-', label=f'min degree')
    plt.yscale('log')
    plt.xscale('symlog', linthresh=1e-6)
    plt.xlabel('noise')
    plt.ylabel('error')
    plt.xlim(left=-1e-7)
    plt.title(f'Projective synchronization with %{int(miss_rate * 100)} of missing edges, and {n} nodes')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_different_noise_aligned(100, 0.8, 20)
    test_different_noise_not_aligned(100, 0.8, 20)
