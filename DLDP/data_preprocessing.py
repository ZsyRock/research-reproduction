import os
from torchvision import datasets, transforms
from torch.utils import data

def load_dataset(dataset_path, batch_size):
    """加载MNIST数据集.

    加载pytorch库的自带的MNIST数据集，并进行两个预处理操作：
        1. 将图片转换为Tensor对象
        2. 归一化图片的像素值，归一化均值方差为 (0.1307, 0.3081)

    Parameters
    ----------
    dataset_path : str
        数据集路径.

    batch_size : int
        批数据量大小.

    Returns
    -------
    train_loader : torch.utils.data.dataloader.DataLoader
        MNIST训练数据集加载器.

    test_loader : torch.utils.data.dataloader.DataLoader
        MNIST测试数据集加载器.
    """

    # 自动展开 ~ 为用户主目录
    dataset_path = os.path.expanduser(dataset_path)

    # 如果路径不存在，自动创建
    os.makedirs(dataset_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = data.DataLoader(
        datasets.MNIST(
            dataset_path,
            train=True,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = data.DataLoader(
        datasets.MNIST(
            dataset_path,
            train=False,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
