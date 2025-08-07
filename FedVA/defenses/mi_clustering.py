import numpy as np
from sklearn.mixture import GaussianMixture
from defenses.plotting import plot_results


def cluster_step1_GM(dim_reduced_gradients, worker_ids, title="Gaussian Mixture", save_dir="figures"):
    gm = GaussianMixture(n_components=3, random_state=0).fit(dim_reduced_gradients)
    result = gm.predict(dim_reduced_gradients)

    # Identify clusters by size
    idx_0, idx_1, idx_2 = [np.where(result == i)[0] for i in range(3)]
    benign_cluster = np.argmax([len(idx_0), len(idx_1), len(idx_2)])
    benign_idx = worker_ids[np.where(result == benign_cluster)[0]]

    plot_results(dim_reduced_gradients, result, gm.means_, gm.covariances_, 0, title, save_dir=save_dir)
    return benign_idx, result
