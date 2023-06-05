import numpy as np
from synchronization.spanning_tree_synchronization import spanning_tree_sync
from synchronization.projective_synchronization_spectral import projective_synch
from homography_synchronization import delete_info
import synchronization.utils as utils
import matplotlib.pyplot as plt
import itertools


def build_z_projective_tr(n: int, d: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X = np.empty([0, d], dtype=complex)
    X_not_scaled = np.empty([0, d])
    X_b = np.empty([d, 0], dtype=complex)
    X_not_scaled_b = np.empty([d, 0])
    for _ in range(n):
        X_i = np.random.randn(d, d)
        det_xi = np.linalg.det(X_i)
        X_not_scaled = np.concatenate([X_not_scaled, X_i], axis=0)
        X_not_scaled_b = np.concatenate([X_not_scaled_b, np.linalg.inv(X_i)], axis=1)
        X_i_complex = X_i / np.power(det_xi, 1 / d, dtype=complex)
        X = np.concatenate([X, X_i_complex], axis=0)
        X_b = np.concatenate([X_b, np.linalg.inv(X_i_complex)], axis=1)
    A = np.ones((n, n))
    return X @ X_b, A, X_not_scaled, X_not_scaled @ X_not_scaled_b


def test(n: int, sigma: float, miss_rate: float):
    Z, A, X_not_scaled, Z_not_scaled = build_z_projective_tr(n, 4)
    Z += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    Z_not_scaled += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    A = delete_info(A, int(miss_rate * np.sum(np.arange(n - 1))), n)
    U_pr_imag, root = projective_synch(Z, A)
    U_pr_real, _ = projective_synch(Z, A, remove_imag=True, root=root)
    U = spanning_tree_sync(Z_not_scaled, A, root)
    X_scaled = utils.scale_matrices(X_not_scaled, 4, root)
    err_pr_imag = utils.get_error(U_pr_imag, X_scaled, 4, distance_type='angle')
    err_pr_real = utils.get_error(U_pr_real, X_scaled, 4, distance_type='angle')
    err_spanning = utils.get_error(U, X_scaled, 4, distance_type='angle')
    return err_pr_real, err_pr_imag, err_spanning


def get_mean_error(n: int, miss_rate: float, sigma: float, num_repeat: int):
    err_pr_real, err_pr_imag, err_sp = 0, 0, 0
    for _ in range(num_repeat):
        res = test(n, miss_rate=miss_rate, sigma=sigma)
        err_pr_real, err_pr_imag, err_sp = err_pr_real + res[0], err_pr_imag + res[1], err_sp + res[2]
    return err_pr_real / num_repeat, err_pr_imag / num_repeat,  err_sp / num_repeat


def test_different_noise(dimensions: list, miss_rate: float):
    sigmas = np.concatenate([[0], np.logspace(0, 6, 7) * 1e-6], axis=0)
    for n in dimensions:
        print(f'n = {n}')
        errors_pr_real_synch = list()
        errors_pr_imag_synch = list()
        errors_spanning_synch = list()
        for sigma in sigmas:
            err_pr_real, err_pr_imag, err_sp = get_mean_error(n, miss_rate, sigma, 20)
            errors_pr_real_synch.append(err_pr_real)
            errors_pr_imag_synch.append(err_pr_imag)
            errors_spanning_synch.append(err_sp)
        plt.figure()
        plt.plot(sigmas, errors_pr_real_synch, 'o-', label=f'spectral (1) sol, n = {n}')
        plt.plot(sigmas, errors_pr_imag_synch, 'o-', label=f'spectral (2) sol, n = {n}')
        plt.plot(sigmas, errors_spanning_synch, 'o-', label=f'spanning tree, n = {n}')
        plt.yscale('log')
        plt.xscale('symlog', linthresh=1e-6)
        plt.xlabel('noise')
        plt.ylabel('error')
        plt.xlim(left=-1e-7)
        plt.title(f'Projective synchronization with %{int(miss_rate * 100)} of missing edges, and {n} nodes')
        plt.legend()
        plt.show()


def test_missing_info(dimensions: list, sigmas: list):
    incomplete_percent = np.linspace(0, 0.98, 10)
    plt.figure(figsize=(10, 8))
    for n, sigma in itertools.product(dimensions, sigmas):
        print(f'n = {n}, sigma = {sigma}')
        errors_pr_real_synch = list()
        errors_pr_imag_synch = list()
        errors_spanning_synch = list()
        for miss_rate in incomplete_percent:
            err_pr_real, err_pr_imag, err_sp = get_mean_error(n, miss_rate, sigma, 20)
            errors_pr_real_synch.append(err_pr_real)
            errors_pr_imag_synch.append(err_pr_imag)
            errors_spanning_synch.append(err_sp)
        plt.semilogy(incomplete_percent * 100, errors_pr_imag_synch, 'o-', label=f'spectral (2) sol, n = {n}, sigma = {sigma}')
        plt.semilogy(incomplete_percent * 100, errors_pr_real_synch, 'o-', label=f'spectral (1) sol, n = {n}, sigma = {sigma}')
        plt.semilogy(incomplete_percent * 100, errors_spanning_synch, 'o-', label=f'spanning tree, n = {n}, sigma = {sigma}')
    plt.xlabel('% missing data')
    plt.ylabel('error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_different_noise([50, 80, 100], 0.8)
    test_missing_info([100], [1e-3])
