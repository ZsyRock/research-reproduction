from .dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from .data_distribution.noniid_dirichlet import distribute_noniid_dirichlet
from .data_distribution.iid_equal import distribute_batches_equally

class FashionMNISTDataset(Dataset):

    def __init__(self, args):
        super(FashionMNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST train data")

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        full_dataset = datasets.FashionMNIST(
            root=self.get_args().get_data_path(),
            train=True,
            download=True,
            transform=transform
        )

        full_loader = DataLoader(full_dataset, batch_size=self.get_args().get_batch_size(), shuffle=True)

        if self.get_args().get_data_distribution_strategy() == "noniid":
            alpha = getattr(self.args, "noniid_alpha", 0.5)
            distributed = distribute_noniid_dirichlet(
                full_loader,
                num_workers=self.get_args().get_num_workers(),
                alpha=alpha
            )
            data, labels = [], []
            for worker_data in distributed:
                for x, y in worker_data:
                    data.append(x.cpu().numpy())
                    labels.append(y.cpu().numpy())
            train_data = (np.stack(data), np.array(labels))
        else:
            train_data = self.get_tuple_from_data_loader(
                DataLoader(full_dataset, batch_size=len(full_dataset))
            )


        self.get_args().get_logger().debug("Finished loading Fashion MNIST train data")
        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST test data")

        test_dataset = datasets.FashionMNIST(
            root=self.get_args().get_data_path(),
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")
        return test_data
