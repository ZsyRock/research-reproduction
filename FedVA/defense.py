"""
defense.py（模块化重构）
- 保留为单文件，但拆分为逻辑模块块结构。
- 适配 mp.py / server.py，避免运行错误。
"""

# ====== 通用导入 ======
import os
import math
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial import Voronoi, voronoi_plot_2d

# ====== 项目内部导入 ======
from client import Client
from federated_learning.arguments import Arguments
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from federated_learning.utils import (
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)


# ====== 全局参数（建议移入 config）======
LAYER_NAME = "fc2.weight"
CLASS_NUM = 1
THRESHOLD = 0.8
POISONED_WORKER_IDS = [31, 12, 30, 14, 45, 39, 10, 15, 33, 24]
SAVE_NAME = "defense_results.jpg"
SAVE_SIZE = (18, 14)

# ====== Part 1: 加载模型工具 ======
def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients

def flatten_layers(model_param):
    p = []
    for layer in model_param:
        param = model_param[layer]
        if hasattr(param, "cpu"):
            p.append(param.cpu().numpy().flatten())
        else:
            p.append(param.flatten())
    return np.concatenate(p, axis=0)

# ====== Part 2: 绘图函数 ======
def plot_gradients_2d(gradients, num_cls=2, marker_list=[POISONED_WORKER_IDS], label=["class_1", "other"], save_name=SAVE_NAME, save_dir="figures"):
    fig = plt.figure()
    size = len(gradients)
    pt = []
    class_array = [np.zeros([len(marker_list[i]), 2]) for i in range(num_cls - 1)]
    class_array.append(np.zeros([size - sum(len(m) for m in marker_list), 2]))
    index = [0] * num_cls
    for (worker_id, gradient) in gradients:
        ploted = False
        for i in range(num_cls - 1):
            if worker_id in marker_list[i]:
                class_array[i][index[i]] = gradient[:2]
                index[i] += 1
                ploted = True
                break
        if not ploted:
            class_array[-1][index[-1]] = gradient[:2]
            index[-1] += 1
    colors = ['b', 'c', 'y', 'm', 'r']
    for i in range(num_cls):
        sc = plt.scatter(class_array[i][:, 0], class_array[i][:, 1], color=colors[i], s=100, label=label[i])
        pt.append(sc)
    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.margins(0.3, 0.3)
    plt.title(save_name)
    plt.legend(pt, label)
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# ====== Part 3: 聚类方法 - 单簇 / 双簇 / 三簇 ======
def cluster_step1_GM(dim_reduced_gradients, worker_ids, title='Gaussian Mixture-1', save_dir="figures"):
    gm = GaussianMixture(n_components=3, random_state=0).fit(dim_reduced_gradients)
    result = gm.predict(dim_reduced_gradients)
    cluster_center_fr = gm.means_[np.argmax([sum(result == i) for i in range(3)])]
    cov_fr = gm.covariances_[np.argmax([sum(result == i) for i in range(3)])] / 1.5
    fr_worker = index_in_fr(dim_reduced_gradients, fr_mean=cluster_center_fr, fr_cov=cov_fr)
    benign_idx = np.asarray(worker_ids)[np.where(result == np.argmax([sum(result == i) for i in range(3)]))[0]]
    other_clusters = [i for i in range(3) if i != np.argmax([sum(result == i) for i in range(3)])]
    dists = [np.linalg.norm(gm.means_[idx] - cluster_center_fr) for idx in other_clusters]
    bad_idx = np.asarray(worker_ids)[np.where(result == other_clusters[np.argmax(dists)])[0]]
    grey_idx = np.asarray(worker_ids)[np.where(result == other_clusters[np.argmin(dists)])[0]]
    fr_idx = np.asarray(worker_ids)[fr_worker]
    plot_results(dim_reduced_gradients, result, gm.means_, gm.covariances_, 0, title, save_dir)
    return benign_idx, bad_idx, grey_idx, fr_idx

def one_cluster(dim_reduced_gradients, worker_ids, fed_pct=0.2, grey_pct=0.2, title='Gaussian Mixture one cluster', save_dir="figures"):
    gm = GaussianMixture(n_components=1, random_state=0).fit(dim_reduced_gradients)
    benign_center = gm.means_[0]
    result = gm.predict(dim_reduced_gradients)
    sorted_benign = sorted([(g, w, np.linalg.norm(g - benign_center)) for g, w in zip(dim_reduced_gradients, worker_ids)], key=lambda x: x[2])
    fed_count = int(len(sorted_benign) * fed_pct)
    grey_count = int(len(sorted_benign) * grey_pct)
    fr_idx = np.array([x[1] for x in sorted_benign[:fed_count]])
    grey_idx = np.array([x[1] for x in sorted_benign[-grey_count:]])
    benign_idx = np.array([x[1] for x in sorted_benign if x[1] not in grey_idx])
    bad_idx = []
    for i, label in enumerate([2]*fed_count + [3]*grey_count):
        g = sorted_benign[i][0]
        _id = np.where((dim_reduced_gradients == g).all(axis=1))
        result[_id] = label
    plot_results(dim_reduced_gradients, result, gm.means_, gm.covariances_, 0, title, save_dir)
    return benign_idx, bad_idx, grey_idx, fr_idx

def two_cluster_GM(dim_reduced_gradients, worker_ids, fed_pct=0.2, grey_pct=0.2, title='Gaussian Mixture two cluster', save_dir="figures"):
    worker_ids = np.asarray(worker_ids)
    gm = GaussianMixture(n_components=2, random_state=0).fit(dim_reduced_gradients)
    result = gm.predict(dim_reduced_gradients)

    index0, index1 = np.where(result == 0)[0], np.where(result == 1)[0]
    benign_cluster = 0 if len(index0) > len(index1) else 1
    benign_center = gm.means_[benign_cluster]
    benign_indices = np.where(result == benign_cluster)[0]

    sorted_benign = sorted(
        [(dim_reduced_gradients[i], worker_ids[i], np.linalg.norm(dim_reduced_gradients[i] - benign_center))
         for i in benign_indices],
        key=lambda x: x[2]
    )

    fed_count = int(len(sorted_benign) * fed_pct)
    grey_count = int(len(sorted_benign) * grey_pct)

    fr_idx = np.array([x[1] for x in sorted_benign[:fed_count]])
    grey_idx = np.array([x[1] for x in sorted_benign[-grey_count:]])
    benign_idx = np.array([x[1] for x in sorted_benign if x[1] not in grey_idx])
    bad_idx = worker_ids[np.where(result == 1 - benign_cluster)[0]]

    for i, label in enumerate([2] * fed_count + [3] * grey_count):
        g = sorted_benign[i][0]
        _id = np.where((dim_reduced_gradients == g).all(axis=1))
        result[_id] = label

    plot_results(dim_reduced_gradients, result, gm.means_, gm.covariances_, 0, title, save_dir)
    return benign_idx, bad_idx, grey_idx, fr_idx

# ====== Part 4: 绘图内部函数（私有）======
color_iter = itertools.cycle(["r", "g", "b", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, index, title, save_dir):
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 8))
    pt = []
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        sc = plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 5, color=color, label=f"Cluster-{i}")
        pt.append(sc)
        angle = 180.0 * np.arctan(u[1] / u[0]) / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    plt.title(title)
    plt.grid(True)
    plt.margins(0.3, 0.3)
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{title}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def index_in_fr(dim_reduced_gradients, fr_mean, fr_cov):
    v, w = linalg.eigh(fr_cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    cos_angle = np.cos(np.radians(180. - np.degrees(angle)))
    sin_angle = np.sin(np.radians(180. - np.degrees(angle)))
    xc = dim_reduced_gradients[:, 0] - fr_mean[0]
    yc = dim_reduced_gradients[:, 1] - fr_mean[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct ** 2 / (v[0] / 2.) ** 2) + (yct ** 2 / (v[1] / 2.) ** 2)
    return np.where(rad_cc <= 1)[0]


# ====== Part 5: 可选的 main（仅供调试） ======
if __name__ == '__main__':
    args = Arguments(logger)
    args.log()
    # 示例：用于调试使用，后续建议拆入 test_defense.py 或 notebook
    print("[INFO] defense.py 可被调用。")
