import torch
import numpy as np

def train(model, train_loader, loss_func, epoch, args, accountant):
    """差分隐私随机梯度下降训练.

    执行差分隐私随机梯度下降训练.

    复现结果与Pytorch中的隐私训练库opacus的结果基本一致，但在隐私统计上有所区别.

    opacus采用的隐私统计方法来自于论文：Bu Z, Dong J, Long Q, Su WJ. Deep Learning with Gaussian Differential Privacy. Harv Data Sci Rev. 2020;2020(23):10.1162/99608f92.cfc5dd25. doi: 10.1162/99608f92.cfc5dd25. Epub 2020 Sep 30. PMID: 33251529; PMCID: PMC7695347.

    这里的实现采用的隐私统计方法来自于论文：Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy[C]//Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016: 308-318.

    Parameters
    ----------
    model : torch.nn.Module
        训练模型.

    train_loader : torch.utils.data.dataloader.DataLoader
        训练数据集加载器.
    
    loss_func : torch.nn.modules.loss
        损失函数.

    epoch : int
        当前训练轮次.
    
    args : argparse.Namespace
        相关参数.
    
    accountant : MomentsAccountant
        隐私统计.
    """

    model.train()  # 开启训练模式

    for batch_idx, (X_data, Y_label) in enumerate(train_loader):
        X_data, Y_label = X_data.to(args.device), Y_label.to(args.device)
        batch_data_num = len(X_data)
        batch_loss = 0.0

        batch_data_parameters_grad_dict = {}  # 用于存储批数据计算得到的参数梯度

        for data_idx, (per_data, per_label) in enumerate(zip(X_data, Y_label)):  # 裁剪每个样本计算得的参数梯度，然后累加
            per_data_parameters_grad_dict = {}  # 用于存储每个样本计算得到的参数梯度
            output = model(per_data.unsqueeze(0))  # 由于模型输入只接受批数据，因此为每个样本添加一个维度
            loss = loss_func(output, per_label.unsqueeze(0))  # 由于模型输入只接受批数据，因此为每个标签添加一个维度
            loss.backward()
            batch_loss += loss.item()

            # 计算每个样本计算得到的参数梯度的范数
            model_parameter_grad_norm = 0.0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    model_parameter_grad_norm += (torch.norm(param.grad) ** 2).item()
                    per_data_parameters_grad_dict[name] = param.grad.clone().detach()
                model_parameter_grad_norm = np.sqrt(model_parameter_grad_norm)
            
                for name in per_data_parameters_grad_dict:
                    per_data_parameters_grad_dict[name] /= max(1, model_parameter_grad_norm / args.C)  # 梯度裁剪
                    if name not in batch_data_parameters_grad_dict:
                        batch_data_parameters_grad_dict[name] = per_data_parameters_grad_dict[name]
                    else:
                        batch_data_parameters_grad_dict[name] += per_data_parameters_grad_dict[name]
                for param in model.parameters():
                    param.grad.zero_()  # 梯度清零

        for name in batch_data_parameters_grad_dict:  # 为批数据计算得到的参数梯度加噪，并求平均
            batch_data_parameters_grad_dict[name] += torch.randn(batch_data_parameters_grad_dict[name].shape).to(args.device) * args.C * args.noise_sigma  # 梯度加噪
            batch_data_parameters_grad_dict[name] /= batch_data_num
        
        with torch.no_grad():  # 使用加噪后梯度进行SGD优化
            for name, param in model.named_parameters():
                param -= args.lr * batch_data_parameters_grad_dict[name]
    
        if (batch_idx + 1) % 30 == 0 and (batch_idx + 1) < len(train_loader):  # 每30次梯度下降后，输出一次损失
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(X_data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), batch_loss / len(X_data)))
        elif (batch_idx + 1) == len(train_loader):  # 最后一批数据处理完之后的输出
            print('Train Epoch: {} [{}/{} ({:.0f})%]\tLoss: {:.6f}'.format(
                epoch, len(train_loader.dataset), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), batch_loss / len(X_data)))
        
    # 计算当前隐私损失
    epsilon, delta = accountant.get_privacy_spent(
        args.noise_sigma,
        args.batch_size / len(train_loader.dataset),
        epoch * len(train_loader),
        args.delta)
    print("(epsilon = {:.2f}, delta = {})".format(epsilon, args.delta))
    
    return 

def test(model, test_loader, loss_func, args):
    """模型测试.

    统计当前轮次，模型在测试集上的表现.

    Parameters
    ----------
    model : torch.nn.Module
        训练模型.

    test_loader : torch.utils.data.dataloader.DataLoader
        测试数据集加载器.
    
    loss_func : torch.nn.modules.loss
        损失函数.
    
    args: argparse.Namespace
        相关参数.
    """

    model.eval()  # 开启测试模式
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X_data, Y_label in test_loader:
            X_data, Y_label = X_data.to(args.device), Y_label.to(args.device)
            output = model(X_data)  # 模型预测输出
            test_loss += loss_func(output, Y_label).item() * len(X_data)  # 累加批数据的损失
            pred = output.argmax(dim=1, keepdim=True)  # 根据模型输出的类别向量获得对应的预测标签
            correct += pred.eq(Y_label.reshape(pred.shape)).sum().item()  # 统计分类正确的样本个数

    test_loss /= len(test_loader.dataset)  # 统计模型在测试集上的平均损失

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 