
import argparse
import torch
import os
from models import MLP
import torch.nn as nn
from data_preprocessing import load_dataset
from dp_train import train, test
from accountant import MomentsAccountant

def main():
    # 实验参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size',
                        default=600)
    parser.add_argument('--epochs',
                        type=int,
                        help='epoch number',
                        default=3)
    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=0.1)
    parser.add_argument('--device',
                        type=torch.device,
                        help='learning rate',
                        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--dataset_path',
                        type=str,
                        help='dataset path',
                        default=os.path.expanduser('~/DLDP/data/mnist'))
    parser.add_argument('--in_features',
                        type=int,
                        help='dimen of input date',
                        default=784)
    parser.add_argument('--n_category',
                        type=int,
                        help='num of category',
                        default=10)
    parser.add_argument('--noise_sigma',
                        type=float,
                        help='noise sigma',
                        default=2.0)
    parser.add_argument('--C',
                        type=float,
                        help='clipping threshold',
                        default=4.0)
    parser.add_argument('--delta',
                        type=float,
                        help='break privacy probability',
                        default=1e-5)
    args = parser.parse_args()

    # 神经网络模型、优化器、损失函数及数据集加载器设置
    model = MLP(args.in_features, args.n_category).to(args.device)
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader = load_dataset(args.dataset_path, args.batch_size)

    accountant = MomentsAccountant()

    # 训练&测试
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_func, epoch, args, accountant)
        test(model, test_loader, loss_func, args)
   
    return

if __name__ == '__main__':
    main()