import numpy as np
import torch
from torch.utils.data import DataLoader


def distribute_noniid_dirichlet(train_loader, num_workers=10, alpha=0.5, seed=42):
    """
    使用 Dirichlet 分布进行 Non-IID 数据划分

    :param train_loader: 原始训练集 DataLoader
    :param num_workers: 客户端数量
    :param alpha: Dirichlet 参数，越小越 Non-IID
    :param seed: 随机种子，控制可复现性
    :return: list，长度为 num_workers，每个元素为该客户端的数据列表
    """
    np.random.seed(seed)
    all_data = []
    all_targets = []

    # Step 1: 合并所有数据
    for batch in train_loader:
        data, target = batch
        all_data.append(data)
        all_targets.append(target)

    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)

    classes = torch.unique(all_targets)
    data_per_worker = [[] for _ in range(num_workers)]

    # Step 2: 对每个类，按 Dirichlet 分布分配样本
    for cls in classes:
        idx_cls = (all_targets == cls).nonzero().flatten()
        np.random.shuffle(idx_cls.numpy())

        proportions = np.random.dirichlet(np.repeat(alpha, num_workers))
        proportions = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
        split_idx = np.split(idx_cls.numpy(), proportions)

        for worker_id, indices in enumerate(split_idx):
            for i in indices:
                data_per_worker[worker_id].append((all_data[i], all_targets[i]))

    return data_per_worker
