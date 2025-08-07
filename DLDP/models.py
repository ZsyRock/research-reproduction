import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """多层感知器.

    含有一层隐藏层的全连接神经网络，隐藏层包含1000个神经元，激活函数为ReLU.

    Parameters
    ----------
    in_features : int
        输入层数据向量的维度.

    n_category : int
        数据类别数.
    """

    def __init__(self, in_features, n_category):
        super().__init__()

        self.fc_1 = nn.Linear(in_features, 1000)
        self.fc_2 = nn.Linear(1000, n_category)
    
    def forward(self, x):
        x = x.reshape(len(x), -1)
        x = F.relu(self.fc_1(x), inplace=True)
        x = self.fc_2(x)

        return x