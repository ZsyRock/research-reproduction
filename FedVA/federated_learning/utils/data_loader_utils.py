import numpy
from .label_replacement import apply_class_label_replacement
import os
import pickle
import random
from ..datasets import Dataset

def generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size):
    """
    Generate data loaders from a distributed dataset.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param batch_size: batch size for data loader
    :type batch_size: int
    """
    data_loaders = []
    for worker_training_data in distributed_dataset:
        # 分离 X 和 Y
        X, Y = zip(*worker_training_data)
        X = numpy.stack([x.numpy() for x in X])  # 转为 np.ndarray
        Y = numpy.array([y.item() for y in Y])   # 转为 np.ndarray
        data_loader = Dataset.get_data_loader_from_data(batch_size, X, Y, shuffle=True)
        data_loaders.append(data_loader)

    return data_loaders

def load_train_data_loader(logger, args):
    strategy = args.get_data_distribution_strategy()

    if strategy == 'noniid':
        train_data_loader_path = "data_loaders/cifar10_noniid/train_data_loader.pickle"
    else:
        train_data_loader_path = "data_loaders/cifar10/train_data_loader.pickle"

    if os.path.exists(train_data_loader_path):
        return load_data_loader_from_file(logger, train_data_loader_path)
    else:
        logger.error(f"Couldn't find train data loader stored in file: {train_data_loader_path}")
        raise FileNotFoundError(f"Couldn't find train data loader stored in file: {train_data_loader_path}")

def generate_train_loader(args, dataset):
    train_dataset = dataset.get_train_dataset()
    X, Y = shuffle_data(args, train_dataset)

    return dataset.get_data_loader_from_data(args.get_batch_size(), X, Y)

def load_test_data_loader(logger, args):
    """
    Loads the test data DataLoader object from a file if available.
    """
    strategy = args.get_data_distribution_strategy()

    if strategy == 'noniid':
        test_data_loader_path = "data_loaders/cifar10_noniid/test_data_loader.pickle"
    else:
        test_data_loader_path = "data_loaders/cifar10/test_data_loader.pickle"

    if os.path.exists(test_data_loader_path):
        return load_data_loader_from_file(logger, test_data_loader_path)
    else:
        logger.error(f"Couldn't find test data loader stored in file: {test_data_loader_path}")
        raise FileNotFoundError(f"Couldn't find test data loader stored in file: {test_data_loader_path}")



def load_data_loader_from_file(logger, filename):
    """
    Loads DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param filename: string
    """
    logger.info("Loading data loader from file: {}".format(filename))

    with open(filename, "rb") as f:
        return load_saved_data_loader(f)

def generate_test_loader(args, dataset):
    test_dataset = dataset.get_test_dataset()
    X, Y = shuffle_data(args, test_dataset)

    return dataset.get_data_loader_from_data(args.get_test_batch_size(), X, Y)

def shuffle_data(args, dataset):
    data = list(zip(dataset[0], dataset[1]))
    random.shuffle(data)
    X, Y = zip(*data)
    X = numpy.asarray(X)
    Y = numpy.asarray(Y)

    return X, Y

def load_saved_data_loader(file_obj):
    return pickle.load(file_obj)

def save_data_loader_to_file(data_loader, file_obj):
    pickle.dump(data_loader, file_obj)
