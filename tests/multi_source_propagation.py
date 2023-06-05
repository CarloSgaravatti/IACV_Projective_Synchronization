import numpy as np
from synchronization.spanning_tree_synchronization import spanning_tree_sync
from synchronization.projective_synchronization_spectral import projective_synch
from synchronization.multi_source_propagation import multi_source_propagation
from homography_synchronization import delete_info
from projective_spanning_spectral_comparison import build_z_projective_tr
import synchronization.utils as utils
import matplotlib.pyplot as plt


def add_outliers(Z_scaled: np.ndarray, Z_not_scaled: np.ndarray, A: np.ndarray, num_outliers: int, n: int):
    valid_transformations = np.triu_indices(n)
    valid_transformations = [(i, j) for i in valid_transformations[0] for j in valid_transformations[1] if A[i, j] == 1]
    outliers = np.random.choice(len(valid_transformations), size=num_outliers, replace=False)
    for k in outliers:
        i, j = valid_transformations[k]
        random_homography = np.random.rand(4, 4)
        random_homography_inverse = np.linalg.inv(random_homography)
        Z_not_scaled[4 * i: 4 * (i + 1), 4 * j: 4 * (j + 1)] = random_homography
        Z_not_scaled[4 * j: 4 * (j + 1), 4 * i: 4 * (i + 1)] = random_homography_inverse
        det = np.linalg.det(random_homography)
        Z_scaled[4 * i: 4 * (i + 1), 4 * j: 4 * (j + 1)] = random_homography / np.power(det, 0.25, dtype=complex)
        Z_scaled[4 * j: 4 * (j + 1), 4 * i: 4 * (i + 1)] = random_homography_inverse * np.power(det, 0.25, dtype=complex)
    return Z_scaled, Z_not_scaled


def test(n: int, sigma: float, miss_rate: float, outliers_percent: float):
    Z, A, X_not_scaled, Z_not_scaled = build_z_projective_tr(n, 4)
    Z_not_scaled += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    for i in range(n):
        for j in range(n):
            det_zij = np.power(np.linalg.det(Z_not_scaled[i * 4: (i + 1) * 4, j * 4: (j + 1) * 4]), 0.25, dtype=complex)
            Z[i * 4: (i + 1) * 4, j * 4: (j + 1) * 4] = Z_not_scaled[i * 4: (i + 1) * 4, j * 4: (j + 1) * 4] / det_zij
    num_elements = np.sum(np.arange(n - 1))
    num_to_delete = int(miss_rate * num_elements)
    A = delete_info(A, num_to_delete, n)
    if outliers_percent > 0.0:
        Z, Z_not_scaled = add_outliers(np.copy(Z), np.copy(Z_not_scaled), A, int(outliers_percent * (num_elements - num_to_delete)), n)
    U_msp, max_degree_node = multi_source_propagation(Z_not_scaled, A)
    X_scaled = utils.scale_matrices(X_not_scaled, 4, max_degree_node)
    U_spectral, _ = projective_synch(Z, A, root=max_degree_node)
    U_spanning = spanning_tree_sync(Z_not_scaled, A, root=max_degree_node)
    err_spectral = utils.get_error(U_spectral, X_scaled, 4, distance_type='angle')
    err_msp = utils.get_error(U_msp, X_scaled, 4, distance_type='angle')
    err_spanning = utils.get_error(U_spanning, X_scaled, 4, distance_type='angle')
    return err_msp, err_spectral, err_spanning


def test_different_noise(n: int, miss_rate: float, num_repeat: int, outliers_percent=0.0):
    sigmas = np.concatenate([[0], np.logspace(0, 6, 7) * 1e-6], axis=0)
    errors_spanning = list()
    errors_msp = list()
    errors_spectral = list()
    for sigma in sigmas:
        err_msp, err_spectral, err_spanning = 0, 0, 0
        for _ in range(num_repeat):
            res = test(n, miss_rate=miss_rate, sigma=sigma, outliers_percent=outliers_percent)
            err_msp, err_spectral, err_spanning = err_msp + res[0], err_spectral + res[1], err_spanning + res[2]
        err_msp, err_spectral, err_spanning = err_msp / num_repeat, err_spectral / num_repeat, err_spanning / num_repeat
        errors_msp.append(err_msp)
        errors_spectral.append(err_spectral)
        errors_spanning.append(err_spanning)
    plt.figure()
    plt.plot(sigmas, errors_msp, 'o-', label='MSP')
    plt.plot(sigmas, errors_spectral, 'o-', label='spectral')
    plt.plot(sigmas, errors_spanning, 'o-', label='spanning')
    plt.yscale('log')
    plt.xscale('symlog', linthresh=1e-6)
    plt.xlabel('noise')
    plt.ylabel('error')
    plt.xlim(left=-1e-7)
    plt.title(f'Projective synchronization with %{int(miss_rate * 100)} of missing edges, and {n} nodes')
    plt.legend()
    plt.show()


def test_outliers(n: int, miss_rate: float, num_repeat: int, noise: float):
    outliers_percent = np.linspace(0, 0.2, 10)
    errors_spanning, errors_msp, errors_spectral = list(), list(), list()
    std_spanning, std_msp, std_spectral = list(), list(), list()
    for outliers in outliers_percent:
        print(outliers)
        err_msp, err_spectral, err_spanning = np.array([]), np.array([]), np.array([])
        for _ in range(num_repeat):
            res = test(n, miss_rate=miss_rate, sigma=noise, outliers_percent=outliers)
            err_msp = np.append(err_msp, res[0])
            err_spectral = np.append(err_spectral, res[1])
            err_spanning = np.append(err_spanning, res[2])
        errors_msp.append(np.mean(err_msp))
        errors_spectral.append(np.mean(err_spectral))
        errors_spanning.append(np.mean(err_spanning))
        std_spanning.append(np.std(err_spanning))
        std_msp.append(np.std(err_msp))
        std_spectral.append(np.std(err_spectral))
    outliers_percent = outliers_percent * 100
    plt.figure()
    plt.plot(outliers_percent, errors_msp, 'bo-', label='MSP')
    plt.fill_between(outliers_percent, np.array(errors_msp) - std_msp, np.array(errors_msp) + std_msp, color='b', alpha=0.2)
    plt.plot(outliers_percent, errors_spectral, 'ro-', label='spectral')
    plt.fill_between(outliers_percent, np.array(errors_spectral) - std_spectral, np.array(errors_spectral) + std_spectral, color='r', alpha=0.2)
    plt.plot(outliers_percent, errors_spanning, 'go-', label='spanning')
    plt.fill_between(outliers_percent, np.array(errors_spanning) - std_spanning, np.array(errors_spanning) + std_spanning, color='g', alpha=0.2)
    plt.xlabel('outliers percentage')
    plt.ylabel('error')
    plt.title(f'{n} nodes, {int(miss_rate * 100)}% of missing edges, noise = {noise}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_different_noise(100, 0.8, 20)
    test_outliers(100, 0.5, 20, 1e-4)
    test_outliers(100, 0.8, 30, 1e-4)
