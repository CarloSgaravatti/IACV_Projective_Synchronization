import itertools
import numpy as np
from synchronization.homography_synchronization import homography_synch
from synchronization.spanning_tree_synchronization import spanning_tree_sync
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import synchronization.utils as utils


def build_z(n: int, d: int) -> (np.ndarray, np.ndarray, np.ndarray):
    X = np.empty([0, 3])
    X_b = np.empty([3, 0])
    for _ in range(n):
        X_i = np.random.rand(d, d)
        det_xi = np.linalg.det(X_i)
        X_i /= (np.sign(det_xi) * np.power(np.abs(det_xi), 1 / d))
        X = np.concatenate([X, X_i], axis=0)
        X_b = np.concatenate([X_b, np.linalg.inv(X_i)], axis=1)
    A = np.ones((n, n))
    return X @ X_b, A, X


def delete_info(A: np.ndarray, num_to_delete: int, n: int) -> np.ndarray:
    is_connected = False
    upper_tri_indices = np.triu_indices(n)
    while not is_connected:
        A_modified = np.copy(A)
        A_modified -= np.diag(A.diagonal())  # eliminate diagonal elements
        to_delete = np.random.choice(upper_tri_indices[0].shape[0], size=num_to_delete, replace=False)
        A_modified[upper_tri_indices[0][to_delete], upper_tri_indices[1][to_delete]] = 0
        A_modified = np.triu(A_modified) + np.triu(A_modified).T
        # graph must be connected for the synchronization problem
        is_connected = connected_components(csr_matrix(A_modified), directed=False, return_labels=False) == 1
    return A_modified


def test_n(n: int, d: int, sigma=None, miss_rate=None):
    Z, A, X = build_z(n, d)
    if sigma is not None:
        Z += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    if miss_rate is not None and miss_rate > 0:
        num_to_delete = int(np.sum(np.arange(n - 1)) * miss_rate)
        A = delete_info(A, num_to_delete, n)
    U_hom_synch, root = homography_synch(Z, A)
    U_spanning_synch = spanning_tree_sync(Z, A, root)
    err_hom = 0
    err_spanning = 0
    for i in range(n):
        err_hom += np.linalg.norm((U_hom_synch[3*i:3*(i+1), :] @ X[3*root:3*(root+1), :]) - X[3*i:3*(i+1), :])
        err_spanning += np.linalg.norm((U_spanning_synch[3 * i:3 * (i + 1), :]
                                        @ X[3*root:3*(root+1), :]) - X[3 * i:3 * (i + 1), :])
    return err_hom / n, err_spanning / n


def test_n_v2(n: int, d: int, sigma=None, miss_rate=None):
    Z, A, X = build_z(n, d)
    if sigma is not None:
        Z += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    if miss_rate is not None and miss_rate > 0:
        num_to_delete = int(np.sum(np.arange(n - 1)) * miss_rate)
        A = delete_info(A, num_to_delete, n)
    U_hom_synch, root = homography_synch(Z, A)
    U_spanning_synch = spanning_tree_sync(Z, A, root)
    X_scaled = utils.scale_matrices(X, d, root)
    err_hom = utils.get_error(U_hom_synch, X_scaled, d, distance_type='angle')
    err_spanning = utils.get_error(U_spanning_synch, X_scaled, d, distance_type='angle')
    return err_hom, err_spanning


def test_different_n():
    dimensions = np.logspace(1, 3, 20, dtype=np.int32)
    errors_hom_synch = list()
    errors_spanning_synch = list()
    for n_nodes in dimensions:
        print(f'n = {n_nodes}')
        err_ho, err_sp = 0, 0
        for _ in range(10):
            res = test_n(n_nodes, 3)
            err_ho, err_sp = err_ho + res[0], err_sp + res[1]
        err_ho, err_sp = err_ho / 10, err_sp / 10
        errors_hom_synch.append(err_ho)
        errors_spanning_synch.append(err_sp)
    plt.figure()
    plt.loglog(dimensions, errors_hom_synch, 'o-', label='spectral sol')
    plt.loglog(dimensions, errors_spanning_synch, 'o-', label='spanning tree')
    plt.xlabel('number of nodes')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def test_different_n_v2():
    dimensions = np.logspace(1, 2.7, 15, dtype=np.int32)  # from 10 to 500
    errors_hom_synch = list()
    errors_spanning_synch = list()
    for n_nodes in dimensions:
        print(f'n = {n_nodes}')
        err_ho, err_sp = 0, 0
        for _ in range(20):
            res = test_n_v2(n_nodes, 3, sigma=1e-3)
            err_ho, err_sp = err_ho + res[0], err_sp + res[1]
        err_ho, err_sp = err_ho / 10, err_sp / 10
        errors_hom_synch.append(err_ho)
        errors_spanning_synch.append(err_sp)
    plt.figure()
    plt.semilogx(dimensions, errors_hom_synch, 'o-', label='spectral sol')
    plt.semilogx(dimensions, errors_spanning_synch, 'o-', label='spanning tree')
    plt.xlabel('number of nodes')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def test_different_noise(dimensions: list, miss_rate: float):
    sigmas = np.logspace(0, 6, 5) * 1e-6
    for n in dimensions:
        print(f'n = {n}')
        errors_hom_synch = list()
        errors_spanning_synch = list()
        for sigma in sigmas:
            err_ho, err_sp = 0, 0
            for _ in range(20):
                res = test_n_v2(n, 3, miss_rate=miss_rate, sigma=sigma)
                err_ho, err_sp = err_ho + res[0], err_sp + res[1]
            err_ho, err_sp = err_ho / 10, err_sp / 10
            errors_hom_synch.append(err_ho)
            errors_spanning_synch.append(err_sp)
        plt.figure()
        plt.loglog(sigmas, errors_hom_synch, 'o-', label=f'spectral sol, n = {n}')
        plt.loglog(sigmas, errors_spanning_synch, 'o-', label=f'spanning tree, n = {n}')
        plt.xlabel('noise')
        plt.ylabel('error')
        plt.title(f'Synchronization with %{int(miss_rate * 100)} of missing edge, and {n} nodes')
        plt.legend()
        plt.show()


def test_incomplete_information(dimensions: list, sigmas: list):
    incomplete_percent = np.linspace(0, 0.98, 10)
    plt.figure(figsize=(10, 8))
    for n, sigma in itertools.product(dimensions, sigmas):
        print(f'n = {n}, sigma = {sigma}')
        errors_hom_synch = list()
        errors_spanning_synch = list()
        for miss_rate in incomplete_percent:
            err_ho, err_sp = 0, 0
            for _ in range(20):
                res = test_n_v2(n, 3, sigma, miss_rate)
                err_ho, err_sp = err_ho + res[0], err_sp + res[1]
            err_ho, err_sp = err_ho / 10, err_sp / 10
            errors_hom_synch.append(err_ho)
            errors_spanning_synch.append(err_sp)
        plt.semilogy(incomplete_percent * 100, errors_hom_synch, 'o-', label=f'spectral sol, n = {n}, sigma = {sigma}')
        plt.semilogy(incomplete_percent * 100, errors_spanning_synch, 'o-', label=f'spanning tree, n = {n}, sigma = {sigma}')
    plt.xlabel('% missing data')
    plt.ylabel('error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test_different_n()
    # test_different_n_v2()
    test_different_noise([50, 80, 120], 0.8)
    test_incomplete_information([100], [1e-4, 5e-3, 1e-3])
