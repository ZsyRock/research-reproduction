from .dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from .data_distribution.noniid_dirichlet import distribute_noniid_dirichlet
from .data_distribution.iid_equal import distribute_batches_equally
from collections import Counter

class CIFAR10Dataset(Dataset):

    def __init__(self, args):
        super(CIFAR10Dataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 train data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        full_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(),
                                        train=True, download=True, transform=transform)
        full_loader = DataLoader(full_dataset, batch_size=self.get_args().get_batch_size(), shuffle=True)

        if self.get_args().get_data_distribution_strategy() == "noniid":
            alpha = getattr(self.args, "noniid_alpha", 0.5)
            distributed = distribute_noniid_dirichlet(
                full_loader,
                num_workers=self.get_args().get_num_workers(),
                alpha=alpha
            )

            # ✅ 打印每个客户端的标签分布
            for worker_id, worker_data in enumerate(distributed):
                labels = [int(y) for _, y in worker_data]
                label_count = dict(Counter(labels))
                self.get_args().get_logger().info(f"[NonIID Client #{worker_id}] Label distribution: {label_count}")

            # 转换为 numpy 格式
            data, labels = [], []
            for worker_data in distributed:
                for x, y in worker_data:
                    data.append(x.cpu().numpy())
                    labels.append(y.cpu().numpy())
            train_data = (np.stack(data), np.array(labels))

        else:
            # ✅ IID 分布也加日志（可选但推荐）
            iid_loader = DataLoader(full_dataset, batch_size=len(full_dataset))
            train_data = self.get_tuple_from_data_loader(iid_loader)

            _, all_labels = train_data
            label_count = dict(Counter(all_labels))
            self.get_args().get_logger().info(f"[IID Global] Label distribution: {label_count}")

        self.get_args().get_logger().debug("Finished loading CIFAR10 train data")
        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(),
                                        train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 test data")

        return test_data
