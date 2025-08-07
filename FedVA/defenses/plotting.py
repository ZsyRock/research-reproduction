import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from scipy import linalg

color_iter = itertools.cycle(["r", "g", "b", "gold", "darkorange"])


def plot_results(X, Y_, means, covariances, index, title, save_dir):
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 8))
    pt = []
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        sc = plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 5, color=color)
        pt.append(sc)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=0.5)
        ax.add_artist(ell)
    plt.title(title)
    plt.grid(True)
    plt.legend(pt, [f"Cluster {i}" for i in range(len(means))])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{title}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close()