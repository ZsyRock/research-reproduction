from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
import os
from client import Client
from federated_learning.utils import get_model_files_for_epoch
from federated_learning.utils import get_model_files_for_suffix
from federated_learning.utils import apply_standard_scaler
from federated_learning.utils import get_worker_num_from_model_file_name
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import calculate_parameter_gradients
from federated_learning.parameters import get_layer_parameters
from defense import load_models, cluster_step1_GM, plot_gradients_2d, two_cluster_GM, one_cluster
import copy
import numpy as np

# MODELS_PATH = "./default_models/3000_models"
CLASS_NUM = 1
LAYER_NAME = "fc2.weight"  # used to be "fc.weight"
THRESHOLD = 0.8
DISCARD_THD = 20


def PCA_clustering_selection(args, epoch, fig_save_dir):
    MODELS_PATH = args.get_save_model_folder_path()#add to find the model path
    param_diff = []
    worker_ids = []
    model_files = sorted(os.listdir(MODELS_PATH))
    start_model_files = get_model_files_for_epoch(model_files, epoch)
    start_model_file = get_model_files_for_suffix(
        start_model_files, args.get_epoch_save_start_suffix())[0]
    start_model_file = os.path.join(MODELS_PATH, start_model_file)
    start_model = load_models(args, [start_model_file])[0]

    start_model_layer_param = list(get_layer_parameters(
        start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

    end_model_files = get_model_files_for_epoch(model_files, epoch)
    end_model_files = get_model_files_for_suffix(
        end_model_files, args.get_epoch_save_end_suffix())

    for end_model_file in end_model_files:
        worker_id = get_worker_num_from_model_file_name(end_model_file)
        end_model_file = os.path.join(MODELS_PATH, end_model_file)
        end_model = load_models(args, [end_model_file])[0]

        end_model_layer_param = list(get_layer_parameters(
            end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        gradient = calculate_parameter_gradients(
            logger, start_model_layer_param, end_model_layer_param)
        gradient = gradient.flatten()

        param_diff.append(gradient)
        worker_ids.append(worker_id)

    #logger.info("Gradients shape: ({}, {})".format(
        #len(param_diff), param_diff[0].shape[0]))
    #logger.info("Prescaled gradients: {}".format(str(param_diff)))
    scaled_param_diff, scl = apply_standard_scaler(param_diff)
    #logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))
    dim_reduced_gradients, pca = calculate_pca_of_gradients(
        logger, scaled_param_diff, 2)
    #logger.info("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))

    #logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(
        #len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    # benign_idx, bad_idx, grey_idx, fr_idx = cluster_step1_GM(dim_reduced_gradients, worker_ids, f"Gaussian Mixture {epoch}")
    #benign_idx, bad_idx, grey_idx, fr_idx  = two_cluster_GM(dim_reduced_gradients, worker_ids, 0.2, 0.2, f"Gaussian Mixture {epoch}")
    benign_idx, bad_idx, grey_idx, fr_idx = one_cluster(
        dim_reduced_gradients, worker_ids, 0.1, 0.5, f"Gaussian Mixture {epoch}", save_dir=fig_save_dir)

    return benign_idx.tolist(), bad_idx.tolist(), grey_idx.tolist(), fr_idx.tolist(), list(zip(worker_ids, dim_reduced_gradients)), pca, scl


def get_fr_train_acc(clients, fr_idx):
    _acc = []
    for id in fr_idx:
        _acc.append(clients[id].test_acc)
    return np.average(_acc)


def verify_by_fr(clients, fr_idx, grey_idx):
    all_valid = []
    fr_size = len(fr_idx)
    i = 0
    for gr in grey_idx:
        all_valid.append([])
        for fr in fr_idx:
            param = clients[gr].get_nn_parameters()
            acc = clients[fr].validate(param)
            print("valid accuracy: ", fr, acc, clients[fr].test_acc)
            if acc < clients[fr].test_acc*0.95:
                all_valid[i].append(0)
            else:
                all_valid[i].append(1)
        i += 1
    # fr_size
    i = 0
    for v in all_valid:
        all_valid[i] = (sum(v) >= fr_size/2)  # valid is True
        i += 1
    return all_valid


def class_validation(clients, fr_idx, grey_idx):
    cls_diff_valid, all_cls_valid = [], []
    fr_size = len(fr_idx)
    i = 0
    for gr in grey_idx:
        cls_diff_valid.append([]); all_cls_valid.append([])
        for fr in fr_idx:
            param = clients[gr].get_nn_parameters()
            _diff, all_cls = clients[fr].by_class_validate(param)
            print("by class valid accuracy: ", gr, fr,
                  _diff, clients[fr].class_diff)
            if _diff > clients[fr].class_diff:
                cls_diff_valid[i].append(0)
            else:
                cls_diff_valid[i].append(1)
            if all_cls:
                all_cls_valid[i].append(1)
            else:
                all_cls_valid[i].append(0)
        i += 1
    # fr_size
    i = 0
    for v in cls_diff_valid:
        cls_diff_valid[i] = (sum(v) >= fr_size/2)
        i += 1
    i = 0
    for v in all_cls_valid:
        all_cls_valid[i] = (sum(v) >= fr_size/2)
        i += 1
    return cls_diff_valid, all_cls_valid


def train_subset_of_clients(epoch, args, clients, mal_count, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    # random select workers to train
    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    old_layer_params = copy.deepcopy(
        list(get_layer_parameters(clients[0].get_nn_parameters(), LAYER_NAME)[CLASS_NUM]))
    
    # training
    for client_idx in random_workers:
        if mal_count[client_idx] > DISCARD_THD:
            args.get_logger().info(
                f"Ignoring client: {client_idx} with mal count {mal_count[client_idx]}")
            continue
        args.get_logger().info("Training epoch #{} on client #{}", str(
            epoch), str(clients[client_idx].get_client_index()))
        if clients[client_idx].mal:
            if args.data_poison:
                if args.mal_strat == 'concat':
                    # model boosting
                    clients[client_idx].concat_train(epoch)
                else:
                    # no boosting
                    clients[client_idx].blend_train(epoch)
            else:
                clients[client_idx].blend_train(epoch)
            
            if args.model_poison == "sign":
                clients[client_idx].sign_attack(epoch)
        else:
            clients[client_idx].benign_train(epoch)

    ## defence 
    if args.defence:

        #benign_models, mal_models, grey_models, fr_models, gradiants, pca, scl = PCA_clustering_selection(
            #args=args, epoch=epoch)
        exp_id = args.get_save_model_folder_path().split("_")[0]  # 从 "3023_models" 提取出 "3023"
        fig_save_dir = f"figures/GS_{exp_id}"
        os.makedirs(fig_save_dir, exist_ok=True)
        benign_models, mal_models, grey_models, fr_models, gradiants, pca, scl = PCA_clustering_selection(args=args, epoch=epoch, fig_save_dir=fig_save_dir)
        #args.get_logger().info(
            #f"Clustered models benign: {benign_models}, mal_models: {mal_models}, grey_models: {grey_models}, fr_models {fr_models}")
        #plot_gradients_2d(gradients=gradiants, num_cls=3, marker_list=[
                      #benign_models, grey_models], save_name=f"Clustering_{epoch}.jpg", label=['benign', 'grey', 'mal'])
    # verification
        bc_vil_r, all_cls_vil = class_validation(
            clients, fr_idx=fr_models, grey_idx=grey_models)
        vil_r = verify_by_fr(clients, fr_idx=fr_models, grey_idx=grey_models)
        #args.get_logger().info(f"By class validation result {bc_vil_r}")
        args.get_logger().info(f"Accuracy validation result {vil_r}")


        for ac_v, bc_v, v, gr in zip(all_cls_vil, bc_vil_r, vil_r, grey_models):
            if not ac_v:
                mal_models.append(gr)
                mal_count[gr] += 1
            elif v and bc_v:
                benign_models.append(gr)
            else:
                mal_models.append(gr)
                mal_count[gr] += 1
        #args.get_logger().info(f"Number Malicious logged: {len(vil_r)-sum(vil_r)}")

        discard_list = []
        for i in range(len(mal_count)):
            if mal_count[i] > DISCARD_THD:
                if i in benign_models:
                    benign_models.remove(i)
                    mal_models.append(i)
                discard_list.append(i)
        args.get_logger().info(
            f"Epoch {epoch}: Newly discarded points: {discard_list}")

        plot_gradients_2d(gradients=gradiants, marker_list=[
                      benign_models], save_name=f"Updated groups after validation_{epoch}.jpg", label=['benign',  'mal'],
                      save_dir = f"figures/exp_{args.get_save_model_folder_path().split('_')[0]}"
# add new save_dir parameter to save current figure in a new directory
                      )

        args.get_logger().info(
            f"Updated benign models: {benign_models}, mal_models: {mal_models}")
        # aggregation
        #args.get_logger().info("Averaging client parameters")
        parameters = [clients[bgn_idx].get_nn_parameters()
                    for bgn_idx in benign_models]
    
        new_nn_params = average_nn_parameters(parameters)  
        new_nn_layer_params = list(get_layer_parameters(
            new_nn_params, LAYER_NAME)[CLASS_NUM]) 
        agg_gradient = calculate_parameter_gradients(
            logger, old_layer_params, new_nn_layer_params)
        agg_gradient = agg_gradient.flatten()
        agg_scaled_param_diff = scl.transform([agg_gradient])
        agg_dim_reduced_gradients = pca.transform(agg_scaled_param_diff)
        gradiants.append((-1, agg_dim_reduced_gradients[0]))
        
        unfiltered_parameters = [
            clients[uf_idx].get_nn_parameters() for uf_idx in random_workers]
        unfiltered_nn = average_nn_parameters(unfiltered_parameters)
        uf_nn_layer_params = list(get_layer_parameters(
            unfiltered_nn, LAYER_NAME)[CLASS_NUM])
        uf_agg_gradient = calculate_parameter_gradients(
            logger, old_layer_params, uf_nn_layer_params)
        uf_agg_gradient = uf_agg_gradient.flatten()
        uf_agg_scaled_param_diff = scl.transform([uf_agg_gradient])
        uf_agg_dim_reduced_gradients = pca.transform(uf_agg_scaled_param_diff)
        gradiants.append((-2, uf_agg_dim_reduced_gradients[0]))

        #plot_gradients_2d(gradiants, num_cls=3,  marker_list=[
                    #[-1], [-2]], save_name=f"gradient plot {epoch}.jpg", label=["aggregated_params", "unfiltered_agg_params", "gradients"])
    else:
        parameters = [
            clients[uf_idx].get_nn_parameters() for uf_idx in random_workers]
        new_nn_params = average_nn_parameters(parameters)  
    #  update to all clients
    for client in clients:
        #args.get_logger().info("Updating parameters on client #{}",
                               #str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    return clients[0].test(), random_workers, mal_count


def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(
            Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients


def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    mal_count = [0]*len(clients)
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected, mal_count = train_subset_of_clients(
            epoch, args, clients, mal_count, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
    args.get_logger().info(
        f"Final malicious count: {mal_count}, discard threashold: {DISCARD_THD}")
    return convert_results_to_csv(epoch_test_set_results), worker_selection

# the main function to run the experiment
def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(
        idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.defence = True #add this line to enable defence
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    global DISCARD_THD
    DISCARD_THD = int(args.get_num_epochs()*0.25)
    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(
        train_data_loader, args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(
        distributed_train_dataset)

    poisoned_workers = identify_random_elements(
        args.get_num_workers(), args.get_num_poisoned_workers())
    # if args.data_poison:
    #     distributed_train_dataset = poison_data(
    #         logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)
        
    train_data_loaders = generate_data_loaders_from_distributed_dataset(
        distributed_train_dataset, args.get_batch_size())

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
