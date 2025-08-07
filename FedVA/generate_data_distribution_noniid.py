from loguru import logger
import os
import pathlib
import pickle
import numpy as np

from federated_learning.arguments import Arguments
from federated_learning.datasets import CIFAR10Dataset
from federated_learning.datasets.data_distribution.noniid_dirichlet import distribute_noniid_dirichlet
from federated_learning.datasets.dataset import Dataset


def save_data_loader_to_file(data_loader, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data_loader, f)
    logger.info(f"Saved DataLoader to: {file_path}")


if __name__ == '__main__':
    args = Arguments(logger)
    args.data_distribution_strategy = "noniid"
    args.noniid_alpha = 0.5  # modify alpha as needed (0.5 to 1.0)

    dataset = CIFAR10Dataset(args)

    # Load the full training dataset
    full_dataset = dataset.get_train_dataset()
    full_data_loader = Dataset.get_data_loader_from_data(
        batch_size=args.get_batch_size(), X=full_dataset[0], Y=full_dataset[1], shuffle=True
    )

    # Perform Dirichlet non-IID partitioning
    distributed_dataset = distribute_noniid_dirichlet(
        full_data_loader,
        num_workers=args.get_num_workers(),
        alpha=args.noniid_alpha
    )

    # Output label distribution (optional)
    for client_id, data in enumerate(distributed_dataset):
        labels = [label.item() for _, label in data]
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"[Client #{client_id}] Label distribution: {dict(zip(unique, counts))}")

    # Save as pickle (an aggregated DataLoader object)
    from federated_learning.utils.data_loader_utils import generate_data_loaders_from_distributed_dataset
    data_loaders = generate_data_loaders_from_distributed_dataset(distributed_dataset, args.get_batch_size())

    save_dir = "data_loaders/cifar10_noniid"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    save_data_loader_to_file(data_loaders, os.path.join(save_dir, "train_data_loader.pickle"))

    # For the test set, the same partition is still used
    test_data_loader = dataset.get_test_loader(args.get_test_batch_size())

    save_data_loader_to_file(test_data_loader, os.path.join(save_dir, "test_data_loader.pickle"))

    logger.success("Non-IID saving completed!")
