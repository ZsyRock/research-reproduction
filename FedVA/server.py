# ====== 基础导入 ======
import os
import copy
import torch
import numpy as np
from loguru import logger
from sklearn.feature_selection import mutual_info_regression

# ====== 项目内部模块导入 ======
from client import Client
from federated_learning.arguments import Arguments
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import (
    generate_data_loaders_from_distributed_dataset,
    average_nn_parameters,
    convert_distributed_data_into_numpy,
    identify_random_elements,
    save_results,
    load_train_data_loader,
    load_test_data_loader,
    generate_experiment_ids,
    convert_results_to_csv,
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from defense import (
    load_models, cluster_step1_GM, plot_gradients_2d,
    two_cluster_GM, one_cluster
)

# ====== 全局参数 ======
CLASS_NUM = 1
#LAYER_NAME = "fc2.weight"
THRESHOLD = 0.8
DISCARD_THD = 20

# ===== Part 1: Flatten 工具函数 =====
def flatten_layers(model_param):
    p = []
    for layer in model_param:
        param = model_param[layer]
        if isinstance(param, torch.Tensor):
            p.append(param.cpu().numpy().flatten())
        else:
            p.append(param.flatten())
    return np.concatenate(p, axis=0)

# ===== Part 2: 主调度入口函数 =====
def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx, config_modifier=None):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(
        idx, 1)
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    if config_modifier is not None:
        args = config_modifier(args)

    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    global DISCARD_THD
    DISCARD_THD = int(args.get_num_epochs() * 0.25)

    if args.get_data_distribution_strategy() == "noniid":
        distributed_train_dataset = train_data_loader  # 已经是 list[DataLoader]
    else:
        from federated_learning.datasets.data_distribution.iid_equal import distribute_batches_equally
        distributed_train_dataset = distribute_batches_equally(
            train_data_loader,
            args.get_num_workers()
        ) ## noniid 分布时不需要转换
    distributed_train_dataset = convert_distributed_data_into_numpy(
        distributed_train_dataset)

    poisoned_workers = identify_random_elements(
        args.get_num_workers(), args.get_num_poisoned_workers())

    if args.get_data_distribution_strategy() == "noniid":
        train_data_loaders = train_data_loader  # 已是 list of DataLoader
    else:
        from federated_learning.datasets.data_distribution.iid_equal import distribute_batches_equally
        train_data_loaders = distribute_batches_equally(
            train_data_loader,
            args.get_num_workers()
        ) ## noniid 分布时不需要转换

    if args.get_data_distribution_strategy() == "noniid":
        for client_id, loader in enumerate(train_data_loaders):
            label_counter = {}
            for _, target in loader:
                for label in target.numpy():
                    label_counter[label] = label_counter.get(label, 0) + 1
            logger.info(f"[Run-time Check] Client #{client_id} label distribution: {label_counter}") ## logging label分布

    clients = create_clients(args, train_data_loaders, test_data_loader)
    for id in range(len(clients)):
        if clients[id].client_idx in poisoned_workers:
            clients[id].mal = True
            if args.data_poison:
                clients[id].poison_data(replacement_method)

    results, worker_selection = run_machine_learning(
        clients, args, poisoned_workers)
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)

# ===== Part 3: 创建客户端对象 =====
def create_clients(args, train_data_loaders, test_data_loader):
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))
    return clients

# ===== Part 4: 联邦训练主循环 =====
def run_machine_learning(clients, args, poisoned_workers):
    epoch_test_set_results = []
    worker_selection = []
    mal_count = [0] * len(clients)
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected, mal_count = train_subset_of_clients(
            epoch, args, clients, mal_count, poisoned_workers)
        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
    args.get_logger().info(
        f"Final malicious count: {mal_count}, discard threashold: {DISCARD_THD}")
    return convert_results_to_csv(epoch_test_set_results), worker_selection

# ===== Part 5: 每轮客户端训练和防御判断 =====
def train_subset_of_clients(epoch, args, clients, mal_count, poisoned_workers):
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())), poisoned_workers, kwargs)

    layer_name = args.layer_name
    old_layer_params = copy.deepcopy(
        list(get_layer_parameters(clients[0].get_nn_parameters(), layer_name)[CLASS_NUM]))


    # 训练
    for client_idx in random_workers:
        if mal_count[client_idx] > DISCARD_THD:
            continue
        if clients[client_idx].mal:
            if args.data_poison:
                if args.mal_strat == 'concat':
                    clients[client_idx].concat_train(epoch)
                else:
                    clients[client_idx].blend_train(epoch)
            else:
                clients[client_idx].blend_train(epoch)
            if args.model_poison == "sign":
                clients[client_idx].sign_attack(epoch)
        else:
            clients[client_idx].benign_train(epoch)

    # === 防御 ===
    if args.defence:
        exp_id = args.get_save_model_folder_path().split("_")[0]
        fig_save_dir = f"figures/GS_{exp_id}"
        os.makedirs(fig_save_dir, exist_ok=True)

        if args.defence == "PCA":
            benign_models, mal_models, grey_models, fr_models, gradiants, pca, scl = PCA_clustering_selection(args, epoch, fig_save_dir)
        elif args.defence == "MI":
            benign_models, mal_models, grey_models, fr_models = mutual_info_clustering_selection(args, epoch)
            gradiants = []
            pca = scl = None
        else:
            raise ValueError(f"Unknown defence method: {args.defence}")

        cls_check, acc_check = class_validation(clients, fr_models, grey_models)
        val_check = verify_by_fr(clients, fr_models, grey_models)

        for gr, cval, aval, vval in zip(grey_models, cls_check, acc_check, val_check):
            if not cval or not aval or not vval:
                mal_models.append(gr)
                mal_count[gr] += 1
            else:
                benign_models.append(gr)

        discard_list = []
        for i, count in enumerate(mal_count):
            if count > DISCARD_THD:
                if i in benign_models:
                    benign_models.remove(i)
                mal_models.append(i)
                discard_list.append(i)

        if gradiants:
            plot_gradients_2d(gradients=gradiants, marker_list=[benign_models], save_name=f"Updated_Epoch_{epoch}.jpg",
                            label=['benign', 'mal'], save_dir=fig_save_dir)

        parameters = [clients[i].get_nn_parameters() for i in benign_models]
    else:
        parameters = [clients[i].get_nn_parameters() for i in random_workers]

    new_nn_params = average_nn_parameters(parameters)
    for client in clients:
        client.update_nn_parameters(new_nn_params)

    return clients[0].test(), random_workers, mal_count

# ===== Part 6: PCA聚类分析防御方法 =====
def PCA_clustering_selection(args, epoch, fig_save_dir):
    MODELS_PATH = args.get_save_model_folder_path()
    model_files = sorted(os.listdir(MODELS_PATH))
    param_diff, worker_ids = [], []

    start_file = get_model_files_for_suffix(get_model_files_for_epoch(model_files, epoch), args.get_epoch_save_start_suffix())[0]
    start_model = load_models(args, [os.path.join(MODELS_PATH, start_file)])[0]
    layer_name = args.layer_name
    start_layer = list(get_layer_parameters(start_model.get_nn_parameters(), layer_name)[CLASS_NUM])

    end_files = get_model_files_for_suffix(get_model_files_for_epoch(model_files, epoch), args.get_epoch_save_end_suffix())
    for f in end_files:
        worker_id = get_worker_num_from_model_file_name(f)
        end_model = load_models(args, [os.path.join(MODELS_PATH, f)])[0]
        end_layer = list(get_layer_parameters(end_model.get_nn_parameters(), layer_name)[CLASS_NUM])
        gradient = calculate_parameter_gradients(logger, start_layer, end_layer).flatten()
        param_diff.append(gradient)
        worker_ids.append(worker_id)

    scaled_diff, scaler = apply_standard_scaler(param_diff)
    dim_reduced, pca = calculate_pca_of_gradients(logger, scaled_diff, 2)

    benign, mal, grey, fr = one_cluster(dim_reduced, worker_ids, 0.1, 0.5, f"GaussianMixture_E{epoch}", save_dir=fig_save_dir)
    return list(benign), list(mal), list(grey), list(fr), list(zip(worker_ids, dim_reduced)), pca, scaler


# ===== Part 7: Mutual Information 防御方法 =====
def mutual_info_clustering_selection(args, epoch):
    MODELS_PATH = args.get_save_model_folder_path()
    model_files = sorted(os.listdir(MODELS_PATH))

    start_file = get_model_files_for_suffix(get_model_files_for_epoch(model_files, epoch), args.get_epoch_save_start_suffix())[0]
    start_model = load_models(args, [os.path.join(MODELS_PATH, start_file)])[0]
    start_vec = flatten_layers(start_model.get_nn_parameters())

    end_files = get_model_files_for_suffix(get_model_files_for_epoch(model_files, epoch), args.get_epoch_save_end_suffix())
    mi_scores, worker_ids = [], []

    for f in end_files:
        worker_id = get_worker_num_from_model_file_name(f)
        end_model = load_models(args, [os.path.join(MODELS_PATH, f)])[0]
        end_vec = flatten_layers(end_model.get_nn_parameters())
        score = mutual_info_regression(start_vec.reshape(-1, 1), end_vec)[0]
        mi_scores.append(score)
        worker_ids.append(worker_id)

    sorted_mi = sorted(zip(mi_scores, worker_ids), key=lambda x: x[0], reverse=True)
    fed_pct, grey_pct = args.fed_pct, args.grey_pct
    fed_count = int(len(sorted_mi) * fed_pct)
    grey_count = int(len(sorted_mi) * grey_pct)

    fed_idx = [x[1] for x in sorted_mi[:fed_count]]
    benign_idx = [x[1] for x in sorted_mi[:-grey_count]]
    grey_idx = [x[1] for x in sorted_mi[-grey_count:]]
    bad_idx = []

    return benign_idx, bad_idx, grey_idx, fed_idx

# ===== Part 8: 联邦保留验证方法 =====
def verify_by_fr(clients, fr_idx, grey_idx):
    results = []
    for g in grey_idx:
        votes = 0
        for f in fr_idx:
            acc = clients[f].validate(clients[g].get_nn_parameters())
            if acc >= clients[f].test_acc * 0.95:
                votes += 1
        results.append(votes >= len(fr_idx) / 2)
    return results

def class_validation(clients, fr_idx, grey_idx):
    cls_pass, acc_pass = [], []
    for g in grey_idx:
        cls_votes, acc_votes = 0, 0
        for f in fr_idx:
            diff, all_cls = clients[f].by_class_validate(clients[g].get_nn_parameters())
            if diff <= clients[f].class_diff:
                cls_votes += 1
            if all_cls:
                acc_votes += 1
        cls_pass.append(cls_votes >= len(fr_idx) / 2)
        acc_pass.append(acc_votes >= len(fr_idx) / 2)
    return cls_pass, acc_pass
