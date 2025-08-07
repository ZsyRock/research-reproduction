# ====== Imports ======
import os
import math
import copy
import torch
import numpy as np
import torch.optim as optim
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from federated_learning.schedulers import MinCapableStepLR
from federated_learning.utils.label_replacement import apply_class_label_replacement

# ====== Client Class ======
class Client:
    def __init__(self, args, client_idx, train_data_loader, test_data_loader):
        self.args = args
        self.client_idx = client_idx
        self.mal = False
        self.device = self.initialize_device()
        self.set_net(self.load_default_model())
        if self.client_idx == 0:
            print(f"[Client #{self.client_idx}] Model architecture:\n{self.net}")



        self.loss_function = self.args.get_loss_function()()
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.args.get_learning_rate(),
            momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.mal_data_loader = None

        # ==== 安全初始化，用于防止 early epoch 报错 ====
        self.test_acc = 0.0
        self.class_diff = 0.0

        # 推断类别数：从 test_data_loader 拿一批样本，计算标签类别数
        if test_data_loader is not None:
            all_labels = []
            for _, labels in test_data_loader:
                all_labels.extend(labels.tolist())
            num_classes = len(set(all_labels))
        else:
            num_classes = 10


        assert num_classes > 0, f"[Client #{self.client_idx}] Failed to infer the number of categories, maybe test_data_loader is empty or does not contain valid labels"
        self.every_class_acc = np.zeros(num_classes)


    # ===== Device Init =====
    def initialize_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() and self.args.get_cuda() else "cpu")

    # ===== NN 操作接口 =====
    def set_net(self, net):
        self.net = net.to(self.device)

    def load_default_model(self):
        model_class = self.args.get_net()
        model = model_class()
        return model


    def load_model_from_file(self, path):
        model_class = self.args.get_net()
        model = model_class()
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path))
            except:
                model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        else:
            self.args.get_logger().warning(f"Model not found: {path}")
        return model

    def get_client_index(self): return self.client_idx
    def get_nn_parameters(self): return copy.deepcopy(self.net.state_dict())
    def update_nn_parameters(self, new_params): self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    # ===== Training Core =====
    def train(self, epoch, dl_type, batch_idx=None):
        self.net.train()
        data_loader = self.mal_data_loader if "mal" in dl_type else self.train_data_loader
        if data_loader is None: return 0.0
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            if batch_idx is not None and i != batch_idx: continue
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if batch_idx is not None: break
        return running_loss

    def benign_train(self, epoch):
        assert not self.mal
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())
        self.train(epoch, "benign")
        self.scheduler.step()
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())
        acc, _, by_class, _ = self.test(self.test_data_loader, log_result=True, epoch=epoch)
        self.test_acc = acc
        self.every_class_acc = np.nan_to_num(by_class)
        self.class_diff = max(by_class) - min(by_class)

    def blend_train(self, epoch):
        assert self.mal
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())
        self.train(epoch, "benign")
        self.scheduler.step()
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())
        acc, _, by_class, _ = self.test(self.test_data_loader, log_result=True, epoch=epoch)
        self.test_acc = acc
        self.every_class_acc = np.nan_to_num(by_class)
        self.class_diff = max(by_class) - min(by_class)

    def sign_attack(self, epoch):
        assert self.mal and self.args.model_poison == 'sign'
        params = self.get_nn_parameters()
        new_params = OrderedDict({k: -v * self.args.sign_scale for k, v in params.items()})
        self.update_nn_parameters(new_params)
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

    def concat_train(self, epoch):
        assert "concat" in self.args.mal_strat and self.mal
        def sub(a, b): return OrderedDict({k: b[k] - a[k] for k in a})
        def add(a, b): return OrderedDict({k: a[k] + b[k] for k in a})
        def scale(d, s): return OrderedDict({k: v * s for k, v in d.items()})
        steps = len(self.train_data_loader) if self.train_data_loader else 1
        for i in range(steps):
            base = self.get_nn_parameters()
            self.train(epoch, "benign", i)
            benign_delta = sub(base, self.get_nn_parameters())
            self.update_nn_parameters(base)
            if self.test(self.mal_data_loader)[1] > 0.0:
                self.train(epoch, "mal")
                mal_delta = sub(base, self.get_nn_parameters())
                boosted = add(benign_delta, scale(mal_delta, self.args.mal_boost))
                self.update_nn_parameters(add(base, boosted))
            else:
                self.update_nn_parameters(add(base, benign_delta))
        self.scheduler.step()
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

    # ===== Poison Data =====
    def poison_data(self, replacement_method):
        assert self.mal and self.args.data_poison
        x, y = [], []
        for data, label in self.train_data_loader:
            x.append(data); y.append(label)
        data = torch.cat(x); label = torch.cat(y)
        total = data.shape[0]
        perm = torch.randperm(total)
        mal_size = int(self.args.poison_ratio * total)
        mal_x, mal_y = apply_class_label_replacement(data[perm[:mal_size]], label[perm[:mal_size]], replacement_method)
        if self.args.mal_strat == 'concat':
            if mal_x.size(0): self.mal_data_loader = DataLoader(TensorDataset(mal_x, mal_y), batch_size=self.args.get_batch_size(), shuffle=True)
            if perm[mal_size:].size(0): self.train_data_loader = DataLoader(TensorDataset(data[perm[mal_size:]], label[perm[mal_size:]]), batch_size=self.args.get_batch_size(), shuffle=True)
        else:
            full_x = torch.cat([mal_x, data[perm[mal_size:]]])
            full_y = torch.cat([mal_y, label[perm[mal_size:]]])
            shuff = torch.randperm(full_x.size(0))
            self.train_data_loader = DataLoader(TensorDataset(full_x[shuff], full_y[shuff]), batch_size=self.args.get_batch_size(), shuffle=True)

    # ===== Evaluation =====
    def test(self, validation_set=None, log_result=False, epoch=None):
        self.net.eval()
        correct, total, loss = 0, 0, 0.0
        pred_, targ_ = [], []
        data_loader = self.test_data_loader if validation_set is None else validation_set
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.net(x)
                _, pred = torch.max(out.data, 1)
                total += y.size(0); correct += (pred == y).sum().item()
                pred_.extend(pred.cpu().numpy()); targ_.extend(y.cpu().numpy())
                loss += self.loss_function(out, y).item()
        acc = 100 * correct / total
        cm = confusion_matrix(targ_, pred_)
        prec = self.calculate_class_precision(cm)
        recall = self.calculate_class_recall(cm)
        if log_result: self.args.get_logger().info(f"[Client #{self.client_idx} Epoch {epoch}] Test Acc: {acc:.2f}% ({correct}/{total})")
        return acc, loss, prec, recall

    def calculate_class_precision(self, cm):
        denom = np.sum(cm, axis=0); denom[denom == 0] = 1
        return np.diagonal(cm) / denom

    def calculate_class_recall(self, cm):
        denom = np.sum(cm, axis=1); denom[denom == 0] = 1
        return np.diagonal(cm) / denom

    # ===== Validation / Comparison =====
    def validate(self, params):
        old = self.get_nn_parameters()
        self.update_nn_parameters(params)
        acc = self.test()[0]
        self.update_nn_parameters(old)
        return acc

    def by_class_validate(self, params):
        old = self.get_nn_parameters()
        self.update_nn_parameters(params)
        by_class = np.nan_to_num(self.test()[2])
        diff = max(by_class) - min(by_class)
        pass_count = (by_class > self.every_class_acc * self.args.cls_thd).sum()
        self.update_nn_parameters(old)
        return diff, pass_count > int(self.args.cls_pass_ratio * len(by_class))

    # ===== Save Model =====
    def save_model(self, epoch, suffix):
        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.makedirs(self.args.get_save_model_folder_path())
        save_path = os.path.join(self.args.get_save_model_folder_path(), f"model_{self.client_idx}_{epoch}_{suffix}.model")
        torch.save(self.get_nn_parameters(), save_path)
