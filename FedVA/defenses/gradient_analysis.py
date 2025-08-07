import os
import numpy as np
from federated_learning.utils import apply_standard_scaler
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from client import Client


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients


def extract_gradients(args, model_files, model_path, epochs, layer_name, class_num, logger):
    param_diff = []
    worker_ids = []
    for epoch in epochs:
        from federated_learning.utils import get_model_files_for_epoch, get_model_files_for_suffix, get_worker_num_from_model_file_name

        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model = load_models(args, [os.path.join(model_path, start_model_file)])[0]
        start_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), layer_name)[class_num])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model = load_models(args, [os.path.join(model_path, end_model_file)])[0]
            end_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), layer_name)[class_num])

            gradient = calculate_parameter_gradients(logger, start_layer_param, end_layer_param).flatten()
            param_diff.append(gradient)
            worker_ids.append(worker_id)

    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
    scaled_diff, _ = apply_standard_scaler(param_diff)
    dim_reduced, _ = calculate_pca_of_gradients(logger, scaled_diff, 2)

    return dim_reduced, worker_ids