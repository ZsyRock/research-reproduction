import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from federated_learning.schedulers import MinCapableStepLR
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from federated_learning.utils.label_replacement import apply_class_label_replacement
import os
import numpy as np
import copy
import math
from collections import OrderedDict
class Client:

    def __init__(self, args, client_idx, train_data_loader, test_data_loader):
        """
        :param args: experiment arguments
        :type args: Arguments
        :param client_idx: Client index
        :type client_idx: int
        :param train_data_loader: Training data loader
        :type train_data_loader: torch.utils.data.DataLoader
        :param test_data_loader: Test data loader
        :type test_data_loader: torch.utils.data.DataLoader
        """
        self.args = args
        self.client_idx = client_idx
        self.mal = False
        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

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

        self.test_acc = None
        self.every_class_acc = None
        self.class_diff = None 

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.get_cuda():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.

        :param net: torch.nn
        """
        self.net = net
        self.net.to(self.device)

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return copy.deepcopy(self.net.state_dict())

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def train(self, epoch, dl_type, batch_idx = None):
        self.net.train()
        running_loss = 0.0
        if "mal" in dl_type:
            dataloader = self.mal_data_loader
        else:
            dataloader = self.train_data_loader
        if dataloader is None:
            return 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            if batch_idx is not None and i != batch_idx:
                continue
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                #self.args.get_logger().info('[%d, %5d] %s loss: %.3f' % (epoch, i, dl_type, running_loss / self.args.get_log_interval()))
                running_loss = 0.0
            if batch_idx is not None:
                break
        return running_loss

    def benign_train(self, epoch):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """
        assert(self.mal == False and self.mal_data_loader is None)
        self.net.train()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())
        
        running_loss = self.train(epoch, "benign")

        self.scheduler.step()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())
        test_accuracy, _, by_class_prec, _ = self.test(self.test_data_loader)
        self.test_acc = test_accuracy
        self.class_diff = max(by_class_prec) - min(by_class_prec)
        self.every_class_acc = np.nan_to_num(by_class_prec)
        return running_loss
    

    def concat_train(self, epoch):
        assert( ("concat" in self.args.mal_strat) and self.mal )
        # We are saving model not delta
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())
        # benign train iteration
        def sub_state_dict(start: OrderedDict, end: OrderedDict):
            delta = OrderedDict()
            for key in start:
                delta[key] = end[key] - start[key]
            return delta  

        def sum_state_dict(start: OrderedDict, delta: OrderedDict):
            end = OrderedDict()
            for key in start:
                end[key] = start[key] + delta[key]
            return end    

        def mul_state_dict(od: OrderedDict, scale: float):
            end = OrderedDict()
            for key in od:
                end[key] = od[key] * scale 
            return end    
        _batch_idx = len(self.train_data_loader) if self.train_data_loader else 1
        for i in range(_batch_idx):
            weight_step_start = self.get_nn_parameters()
            running_loss = self.train(epoch, "benign", i)
            weight_step_benign = self.get_nn_parameters()

            benign_delta = sub_state_dict(weight_step_start, weight_step_benign)
            #benign_delta = weight_step_benign - weight_step_start

            # mal train iteration
            self.update_nn_parameters(weight_step_start)
            mal_loss_curr = self.test(self.mal_data_loader)[1]
            if mal_loss_curr > 0.0:
                running_loss = self.train(epoch, "mal")
                weight_step_mal = self.get_nn_parameters()
                mal_delta = sub_state_dict(weight_step_start, weight_step_mal)
                _boost = mul_state_dict(mal_delta, self.args.mal_boost)
                overall_delta_step = sum_state_dict(benign_delta, _boost)
                self.update_nn_parameters( sum_state_dict(weight_step_start, overall_delta_step))
            else:
                self.update_nn_parameters( sum_state_dict(weight_step_start, benign_delta))
            #self.args.get_logger().info('Benign: Loss - {}; Mal: Loss - {}'.format
                #(self.test(self.train_data_loader)[1], self.test(self.mal_data_loader)[1]))
        self.scheduler.step()
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())
        
        # Question: what if test_loader is manipulated?
        test_accuracy, _, by_class_prec, _ = self.test(self.test_data_loader)
        self.test_acc = test_accuracy
        self.class_diff = max(by_class_prec) - min(by_class_prec)
        self.every_class_acc = np.nan_to_num(by_class_prec)
        return running_loss
    
    def blend_train(self, epoch):
        assert(self.mal == True)
        self.net.train()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())
        
        running_loss = self.train(epoch, "benign")

        self.scheduler.step()

        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())
        test_accuracy, _, by_class_prec, _ = self.test(self.test_data_loader)
        self.test_acc = test_accuracy
        self.class_diff = max(by_class_prec) - min(by_class_prec)
        self.every_class_acc = np.nan_to_num(by_class_prec)
        return running_loss

    def sign_attack(self,epoch):
        assert(self.mal == True)
        assert(self.args.model_poison == 'sign')
        # save model
        params = self.get_nn_parameters()
        new_params = OrderedDict()
        for key in params.keys():
                new_params[key] = -params[key] * self.args.sign_scale
        
        self.update_nn_parameters(new_params)
        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())
        test_accuracy, _, by_class_prec, _ = self.test(self.test_data_loader)
        self.test_acc = test_accuracy
        self.class_diff = max(by_class_prec) - min(by_class_prec)
        self.every_class_acc = np.nan_to_num(by_class_prec)
        return 


    def save_model(self, epoch, suffix):
        """
        Saves the model if necessary.
        """
        self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.mkdir(self.args.get_save_model_folder_path())

        full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
        torch.save(self.get_nn_parameters(), full_save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=1)

    def test(self, validation_set = None):
        self.net.eval()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        data_loader =  self.test_data_loader if validation_set is None else validation_set
        with torch.no_grad():
            for (images, labels) in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        #self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        #self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        #self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        #self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        #self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall

    def poison_data(self, replacement_method):
        assert(self.mal == True and self.args.data_poison == True)
        data_list, label_list = [], []
        for i , (data, label) in enumerate(self.train_data_loader):
            data_list.append(data); label_list.append(label)
        data = torch.cat(data_list) ; label = torch.cat(label_list)
        total_points = data.shape[0]
        perm = torch.randperm(data.size(0))
        mal_points = int(self.args.poison_ratio * total_points)
        mal_idx, bgn_idx = perm[:mal_points],perm[mal_points:]
        mal_x, mal_y = apply_class_label_replacement(data[mal_idx], label[mal_idx], replacement_method)
        
        if self.args.mal_strat == 'concat':
            if mal_idx.shape[0] != 0:
                mal_dataset = TensorDataset(mal_x, mal_y)
                self.mal_data_loader = DataLoader(mal_dataset, batch_size=self.args.get_batch_size(), shuffle=True)
            else:
                self.mal_data_loader = None
            if bgn_idx.shape[0]!=0:
                bgn_dataset = TensorDataset(data[bgn_idx], label[bgn_idx])
                self.train_data_loader = DataLoader(bgn_dataset, batch_size=self.args.get_batch_size(), shuffle=True)
            else:
                self.train_data_loader = None
        else:
            _shuff = torch.randperm(data.size(0))
            train_data, train_label = torch.cat([mal_x, data[bgn_idx]]),torch.cat([mal_y, label[bgn_idx]])
            self.train_data_loader = DataLoader(TensorDataset(train_data[_shuff], train_label[_shuff]), batch_size=self.args.get_batch_size(), shuffle=True)

    def validate(self, params):
        old_params = copy.deepcopy(self.get_nn_parameters())
        self.update_nn_parameters(params)
        acc = self.test()[0]
        self.update_nn_parameters(old_params)
        return acc
    
    def by_class_validate(self, params):
        old_params = copy.deepcopy(self.get_nn_parameters())
        self.update_nn_parameters(params)
        by_class = self.test()[2]
        print("By class: ", by_class)
        by_class = np.nan_to_num(by_class)
        over_thd = by_class > self.every_class_acc * self.args.cls_thd
        _diff = max(by_class) - min(by_class)
        self.update_nn_parameters(old_params)
        pass_count_thd = math.ceil(self.args.cls_pass_ratio*by_class.shape[0])
        return _diff, np.sum(over_thd)>pass_count_thd
