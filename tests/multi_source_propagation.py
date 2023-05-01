import numpy as np
from synchronization.spanning_tree_synchronization import spanning_tree_sync
from synchronization.projective_synchronization_spectral import projective_synch
from synchronization.multi_source_propagation import multi_source_propagation
from homography_synchronization import delete_info
from projective_spanning_spectral_comparison import build_z_projective_tr, get_mean_error
import synchronization.utils as utils
import matplotlib.pyplot as plt


def test(n: int, sigma: float, miss_rate: float):
    Z, A, X_not_scaled, Z_not_scaled = build_z_projective_tr(n, 4)
    Z += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    Z_not_scaled += np.random.randn(Z.shape[0], Z.shape[1]) * sigma
    A = delete_info(A, int(miss_rate * np.sum(np.arange(n - 1))), n)
    U_msp, max_degree_node = multi_source_propagation(Z_not_scaled, A)
    X_scaled = utils.scale_matrices(X_not_scaled, 4, max_degree_node)
    U_spectral, _ = projective_synch(Z, A, root=max_degree_node)
    U_spanning = spanning_tree_sync(Z_not_scaled, A, root=max_degree_node)
    err_spectral = utils.get_error(U_spectral, X_scaled, 4, distance_type='angle')
    err_msp = utils.get_error(U_msp, X_scaled, 4, distance_type='angle')
    err_spanning = utils.get_error(U_spanning, X_scaled, 4, distance_type='angle')
    return err_msp, err_spectral, err_spanning


def test_different_noise(n: int, miss_rate: float, num_repeat: int):
    sigmas = np.concatenate([[0], np.logspace(0, 6, 7) * 1e-6], axis=0)
    errors_spanning = list()
    errors_msp = list()
    errors_spectral = list()
    for sigma in sigmas:
        err_msp, err_spectral, err_spanning = 0, 0, 0
        for _ in range(num_repeat):
            res = test(n, miss_rate=miss_rate, sigma=sigma)
            err_msp, err_spectral, err_spanning = err_msp + res[0], err_spectral + res[1], err_spanning + res[2]
        err_msp, err_spectral, err_spanning = err_msp / num_repeat, err_spectral / num_repeat, err_spanning / num_repeat
        errors_msp.append(err_msp)
        errors_spectral.append(err_spectral)
        errors_spanning.append(err_spanning)
    plt.figure()
    plt.plot(sigmas, errors_msp, 'o-', label=f'MSP')
    plt.plot(sigmas, errors_spectral, 'o-', label=f'spectral')
    plt.plot(sigmas, errors_spanning, 'o-', label=f'spanning')
    plt.yscale('log')
    plt.xscale('symlog', linthresh=1e-6)
    plt.xlabel('noise')
    plt.ylabel('error')
    plt.xlim(left=-1e-7)
    plt.title(f'Projective synchronization with %{int(miss_rate * 100)} of missing edges, and {n} nodes')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_different_noise(100, 0.8, 20)
