import os
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters
from federated_learning.parameters import calculate_parameter_gradients
from federated_learning.utils import get_model_files_for_epoch
from federated_learning.utils import get_model_files_for_suffix
from federated_learning.utils import apply_standard_scaler
from federated_learning.utils import get_worker_num_from_model_file_name
from client import Client
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
from scipy import linalg
import itertools
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import math

'''
# Paths you need to put in.
MODELS_PATH = "./3000_models"
EXP_INFO_PATH = "./logs/3000.log"
'''
# 自动获取最新实验编号（基于已有的日志或模型文件夹）
def find_latest_exp_idx(base=3000, limit=10000):
    for idx in reversed(range(base, limit)):
        if os.path.exists(f"logs/{idx}.log") and os.path.exists(f"{idx}_models"):
            return idx
    raise FileNotFoundError("No valid experiment index found.")

LATEST_EXP_IDX = find_latest_exp_idx()
SAVE_DIR_GS = os.path.join("figures", f"GS_{LATEST_EXP_IDX}")
os.makedirs(SAVE_DIR_GS, exist_ok=True)
# Paths you need to put in.
MODELS_PATH = f"./{LATEST_EXP_IDX}_models"
EXP_INFO_PATH = f"./logs/{LATEST_EXP_IDX}.log"
print(f"Using models from: {MODELS_PATH}")

# The epochs over which you are calculating gradients.
EPOCHS = list(range(1, 11))

# The layer of the NNs that you want to investigate.
#   If you are using the provided Fashion MNIST CNN, this should be "fc.weight"
#   If you are using the provided Cifar 10 CNN, this should be "fc2.weight"
#LAYER_NAME = "fc.weight2"
LAYER_NAME = "fc2.weight"
# The source class.
CLASS_NUM = 1

# The IDs for the poisoned workers. This needs to be manually filled out.
# You can find this information at the beginning of an experiment's log file.
POISONED_WORKER_IDS = [31, 12, 30, 14, 45, 39, 10, 15, 33, 24]

# The resulting graph is saved to a file
SAVE_NAME = "defense_results.jpg"
SAVE_SIZE = (18, 14)

THRESHOLD = 0.8

def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))

        clients.append(client)

    return clients


def plot_gradients_2d(gradients, num_cls = 2, marker_list =[POISONED_WORKER_IDS], label=["class_1", "other"],  save_name = SAVE_NAME, save_dir="figures"  # ← 新增参数，默认保存到 figures 文件夹
                      ):
    fig = plt.figure()
    size = len(gradients)
    class_array = []
    pt = []
    colors = ['b', 'c', 'y', 'm', 'r']
    index = [0]*num_cls
    for i in range(num_cls-1):
        class_array.append(np.zeros([len(marker_list[i]),2]))
        size-=len(marker_list[i])
    class_array.append(np.zeros([size,2]))
    for (worker_id, gradient) in gradients:
        # KNN, gaussian
        ploted = False
        for i in range(num_cls-1):
            if worker_id in marker_list[i]:
                class_array[i][index[i]][0],class_array[i][index[i]][1] = gradient[0], gradient[1]
                index[i]+=1
                ploted = True
                break
        if not ploted:
            class_array[-1][index[-1]][0],class_array[-1][index[-1]][1] = gradient[0], gradient[1]
            index[-1]+=1
    for i in range(num_cls):
        sc = plt.scatter(class_array[i][:,0], class_array[i][:,1],color=colors[i], s=100, label=label[i])
        pt.append(sc)

    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.margins(0.3,0.3)
    plt.title(save_name)
    plt.legend(pt, label)
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True) # 创建保存目录（如果不存在）
    save_path = os.path.join(save_dir, save_name) # 新增拼接完整保存路径
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def index_in_fr(dim_reduced_gradients, fr_mean, fr_cov):
    v, w = linalg.eigh(fr_cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi 
    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))

    xc = dim_reduced_gradients[:, 0] - fr_mean[0]
    yc = dim_reduced_gradients[:, 1] - fr_mean[1]
    #print(dim_reduced_gradients[:,0])
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 

    rad_cc = (xct**2/(v[0]/2.)**2) + (yct**2/(v[1]/2.)**2)
    return np.where(rad_cc<=1)[0]

def cluster_step1_GM(dim_reduced_gradients, worker_ids, title='Gaussian Mixture-1', save_dir="figures"):
    gm = GaussianMixture(n_components=3, random_state=0).fit(dim_reduced_gradients)
    result = gm.predict(dim_reduced_gradients)

    index0 = np.where(result == 0)[0]
    index1 = np.where(result == 1)[0]
    index2 = np.where(result == 2)[0]
    benign_cluster = np.argmax([len(index0), len(index1), len(index2)])

    cluster_center_fr = gm.means_[benign_cluster]
    cov_fr = gm.covariances_[benign_cluster] / 1.5

    index = [0, 1, 2]
    index.remove(benign_cluster)
    dists = [np.linalg.norm(gm.means_[idx] - cluster_center_fr) for idx in index]
    bad_cluster_idx, grey_cluster_idx = (index[0], index[1]) if dists[0] > dists[1] else (index[1], index[0])

    worker_ids = np.asarray(worker_ids)
    benign_idx = worker_ids[np.where(result == benign_cluster)[0]]
    bad_idx = worker_ids[np.where(result == bad_cluster_idx)[0]]
    grey_idx = worker_ids[np.where(result == grey_cluster_idx)[0]]

    fr_worker = index_in_fr(dim_reduced_gradients, fr_mean=cluster_center_fr, fr_cov=cov_fr)
    fr_idx = worker_ids[fr_worker]

    covs = np.concatenate((gm.covariances_, [cov_fr]))
    means = np.concatenate((gm.means_, [cluster_center_fr]))

    plot_results(dim_reduced_gradients, result, means, covs, 0, title, save_dir=save_dir)
    return benign_idx, bad_idx, grey_idx, fr_idx

   
def one_cluster(dim_reduced_gradients, worker_ids, fed_pct=0.2, grey_pct=0.2,
                title='Gaussian Mixture one cluster', save_dir="figures"):
    worker_ids = np.asarray(worker_ids)
    gm = GaussianMixture(n_components=1, random_state=0).fit(dim_reduced_gradients)
    result = gm.predict(dim_reduced_gradients)
    # assume benign cluster have most points
    # label of benign cluster
    benign_cluster = 0
    # center and cov of fed reserve
    benign_center = gm.means_[benign_cluster]
    benign_count =np.where(result == benign_cluster)[0].shape[0]
    fed_count = int(benign_count*fed_pct)
    grey_count = int(benign_count*grey_pct)

    benign_naive = zip(dim_reduced_gradients[np.where(result == benign_cluster)[0]], worker_ids[np.where(result == benign_cluster)[0]])
    benign_confidence = [(x[0], x[1], np.linalg.norm(x[0]-benign_center)) for x in benign_naive] 
    sorted_benign_confidence =  sorted(benign_confidence, key=lambda x: x[2])
    benign_idx = worker_ids[np.where(result == benign_cluster)[0]]
    bad_cluster_idx = 1
    bad_idx = worker_ids[np.where(result == bad_cluster_idx)[0]]
    fr_idx = np.asarray([x[1] for x in sorted_benign_confidence[:fed_count]])
    grey_idx = np.asarray([x[1] for x in sorted_benign_confidence[-grey_count:]])
    benign_idx = np.asarray([x for x in benign_idx.tolist() if x not in grey_idx.tolist()])
    print("cluster index", benign_idx, grey_idx)
    fr_label, gr_label = 2,3
    for fr in sorted_benign_confidence[:fed_count]:
        print("fr element: ", fr)
        _id = np.where((dim_reduced_gradients == fr[0]).all(axis=1))
        print(_id)
        result[_id] = fr_label

    for gr in sorted_benign_confidence[-grey_count:]:
        print("gr element: ", gr)
        _id = np.where((dim_reduced_gradients == gr[0]).all(axis=1))
        print(_id)
        result[_id] = gr_label
    # cluster_center_fr = gm.means_[benign_cluster]
    # cov_fr =gm.covariances_[benign_cluster]/(fed_pct) 

    # find mal and grey cluster
    # print(bad_points, result, bad_cluster_idx)
    covs, means = gm.covariances_, gm.means_
    # exchange to add 2 lines: plot_results(dim_reduced_gradients, result, means, covs, 0, title)
    plot_results(dim_reduced_gradients, result, means, covs, 0, title, save_dir=save_dir)
    return benign_idx, bad_idx, grey_idx, fr_idx


def two_cluster_GM(dim_reduced_gradients, worker_ids, fed_pct=0.2, grey_pct=0.2, title='Gaussian Mixture two cluster', save_dir="figures"):
    worker_ids = np.asarray(worker_ids)
    gm = GaussianMixture(n_components=2, random_state=0).fit(dim_reduced_gradients)
    result = gm.predict(dim_reduced_gradients)

    index0, index1 = np.where(result == 0)[0], np.where(result == 1)[0]
    benign_cluster = 0 if len(index0) > len(index1) else 1
    benign_center = gm.means_[benign_cluster]
    benign_indices = np.where(result == benign_cluster)[0]

    benign_naive = zip(dim_reduced_gradients[benign_indices], worker_ids[benign_indices])
    sorted_benign = sorted([(x[0], x[1], np.linalg.norm(x[0] - benign_center)) for x in benign_naive], key=lambda x: x[2])

    fed_count = int(len(sorted_benign) * fed_pct)
    grey_count = int(len(sorted_benign) * grey_pct)

    benign_idx = [x[1] for x in sorted_benign if x[1] not in [x[1] for x in sorted_benign[-grey_count:]]]
    bad_idx = worker_ids[np.where(result == 1 - benign_cluster)[0]]
    fr_idx = np.array([x[1] for x in sorted_benign[:fed_count]])
    grey_idx = np.array([x[1] for x in sorted_benign[-grey_count:]])

    for fr in sorted_benign[:fed_count]:
        _id = np.where((dim_reduced_gradients == fr[0]).all(axis=1))
        result[_id] = 2
    for gr in sorted_benign[-grey_count:]:
        _id = np.where((dim_reduced_gradients == gr[0]).all(axis=1))
        result[_id] = 3

    plot_results(dim_reduced_gradients, result, gm.means_, gm.covariances_, 0, title, save_dir=save_dir)
    return np.array(benign_idx), bad_idx, grey_idx, fr_idx


color_iter = itertools.cycle(["r", "g", "b", "gold", "darkorange"])

def cluster_step1_KM(dim_reduced_gradients, save_dir="figures"):
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dim_reduced_gradients)
    centers = kmeans.cluster_centers_

    plt.scatter(centers[:, 0], centers[:, 1], marker='s', s=100)
    vor = Voronoi(centers)
    fig = voronoi_plot_2d(vor, plt.gca())
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "Cluster1-2.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close()

   
   
   
color_iter = itertools.cycle(["r", "g", "b", "gold", "darkorange"])

# color_iter = itertools.cycle(["navy", "c", "cornflowerblue","gold"]) # , , "darkorange"
   
def plot_results(X, Y_, means, covariances, index, title, save_dir):
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 8))
    pt = []
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        sc = plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 5, color=color, label=f"Benign")
        pt.append(sc)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    for i, label, color in zip([1, 2, 3], ["mal", "fedres", "grey"], color_iter):
        sc = plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 5, color=color, label=label)
        pt.append(sc)
    plt.title(title)
    plt.grid(True)
    plt.margins(0.3, 0.3)
    plt.legend(pt, ['Benign', "Malicious", 'FR', 'Suspicious'])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{title}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close()



# def cluster_step2_GM():
   


if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    # 自动确定最新实验编号
    exp_id = find_latest_exp_idx()
    save_dir = os.path.join("figures", f"GS_{exp_id}")
    os.makedirs(save_dir, exist_ok=True)

    model_path = f"./{exp_id}_models"
    model_files = sorted(os.listdir(model_path))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(model_path, start_model_file)
        start_model = load_models(args, [start_model_file])[0]
        start_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(model_path, end_model_file)
            end_model = load_models(args, [end_model_file])[0]
            end_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

            gradient = calculate_parameter_gradients(logger, start_layer_param, end_layer_param).flatten()
            param_diff.append(gradient)
            worker_ids.append(worker_id)

    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
    scaled_diff, _ = apply_standard_scaler(param_diff)
    dim_reduced, _ = calculate_pca_of_gradients(logger, scaled_diff, 2)

    # 聚类分析并绘图保存
    benign_idx, bad_idx, grey_idx, fr_idx = cluster_step1_GM(dim_reduced, worker_ids, title="Gaussian Mixture All", save_dir=save_dir)

    sampled_grey_idx = np.random.choice(grey_idx, len(fr_idx))
    fr_clients = fr_idx
    s_grey_params = sampled_grey_idx

    good_list, bad_list = [], []
    for c, param, idx in zip(fr_clients, s_grey_params, sampled_grey_idx):
        acc = 1  # TODO: Replace with actual validation logic
        if acc >= THRESHOLD:
            good_list.append(idx)
        else:
            bad_list.append(idx)

    benign_idx = np.concatenate([benign_idx, good_list])
    bad_idx = np.concatenate([bad_idx, bad_list])
    filtered_idx = np.concatenate([benign_idx, bad_idx]).astype(int)
    labels = np.array([0]*len(benign_idx) + [1]*len(bad_idx))
    X = np.take(dim_reduced, filtered_idx, axis=0)

    gm = GaussianMixture(n_components=2, random_state=0).fit(X, labels)
    cluster_step1_KM(dim_reduced, save_dir=save_dir)  # KMeans图

    logger.info(f"Finished plotting to {save_dir}")
