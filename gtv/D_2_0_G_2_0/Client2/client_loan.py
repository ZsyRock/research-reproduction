import argparse
import torch
import pandas as pd
from torch.nn import Linear
import torch.distributed.rpc as rpc
from torch.autograd import Variable
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autograd
from torch.nn import functional as F
# import imageio
import numpy as np
import os
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.transformer import DataTransformer


def _call_method(method, rref, *args, **kwargs):
    """helper for _remote_method()"""
    return method(rref.local_value(), *args, **kwargs)
def _remote_method(method, rref, *args, **kwargs):
    """
    executes method(*args, **kwargs) on the from the machine that owns rref
    very similar to rref.remote().method(*args, **kwargs), but method() doesn't have to be in the remote scope
    """
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    print(param_rrefs)
    return param_rrefs


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)

def cond_loss(data, output_info, c, m):
    '''
    used to calculate generator loss
    '''
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            continue

        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none')
            loss.append(tmp)
            st = ed
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]

class Discriminator(Module):
    """Discriminator for the CTGANSynthesizer."""

    def __init__(self, input_dim, discriminator_dim):
        super(Discriminator, self).__init__()
        dim = input_dim 
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
        """Compute the gradient penalty."""

        alpha = torch.rand(real_data.size(0), 1, 1, device=device)
        alpha = alpha.repeat(1, 1, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1,  real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
 
        return self.seq(input_)

class Residual(Module):
    """Residual layer for the CTGANSynthesizer."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGANSynthesizer."""

    def __init__(self, embedding_dim, generator_dim, data_dim=256, output_split_list = []):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.output_split_list = output_split_list
        self.data_dim = data_dim

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        assert np.sum(self.output_split_list) == self.data_dim
        output_list = []
        cursor = 0
        for var in self.output_split_list:
            output_list.append(data[:,cursor:cursor+var])
            cursor += var
        return output_list

def random_choice_prob_index_sampling(probs,col_idx):
    
    """
    Used to sample a specific category within a chosen one-hot-encoding representation 

    Inputs:
    1) probs -> probability mass distribution of categories 
    2) col_idx -> index used to identify any given one-hot-encoding
    
    Outputs:
    1) option_list -> list of chosen categories 
    
    """

    option_list = []
    for i in col_idx:
        # for improved stability
        pp = probs[i] + 1e-6 
        pp = pp / sum(pp)
        # sampled based on given probability mass distribution of categories within the given one-hot-encoding 
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

class Condvec(object):
    
    """
    This class is responsible for sampling conditional vectors to be supplied to the generator

    Variables:
    1) model -> list containing an index of highlighted categories in their corresponding one-hot-encoded represenations
    2) interval -> an array holding the respective one-hot-encoding starting positions and sizes     
    3) n_col -> total no. of one-hot-encoding representations
    4) n_opt -> total no. of distinct categories across all one-hot-encoding representations
    5) p_log_sampling -> list containing log of probability mass distribution of categories within their respective one-hot-encoding representations
    6) p_sampling -> list containing probability mass distribution of categories within their respective one-hot-encoding representations

    Methods:
    1) __init__() -> takes transformed input data with respective column information to compute class variables
    2) sample_train() -> used to sample the conditional vector during training of the model
    3) sample() -> used to sample the conditional vector for generating data after training is finished
    
    """


    def __init__(self, data, output_info):
              
        self.model = []
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0 
        self.p_log_sampling = []  
        self.p_sampling = [] 
        
        # iterating through the transformed input data columns 
        st = 0
        for item in output_info:
            # ignoring columns that do not represent one-hot-encodings
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                # using starting (st) and ending (ed) position of any given one-hot-encoded representation to obtain relevant information
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)  
                log_freq = np.log(freq + 1)  
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                pmf = freq / np.sum(freq)
                self.p_sampling.append(pmf)
                st = ed
           
        self.interval = np.asarray(self.interval)
        
    def sample_train(self, batch):
        
        """
        Used to create the conditional vectors for feeding it to the generator during training

        Inputs:
        1) batch -> no. of data records to be generated in a batch

        Outputs:
        1) vec -> a matrix containing a conditional vector for each data point to be generated 
        2) mask -> a matrix to identify chosen one-hot-encodings across the batch
        3) idx -> list of chosen one-hot encoding across the batch
        4) opt1prime -> selected categories within chosen one-hot-encodings

        """

        if self.n_col == 0:
            return None
        batch = batch
        
        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')

        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # matrix of shape (batch x total no. of one-hot-encoded representations) with 1 in indexes of chosen representations and 0 elsewhere
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        
        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_log_sampling,idx) 
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        
        """
        Used to create the conditional vectors for feeding it to the generator after training is finished

        Inputs:
        1) batch -> no. of data records to be generated in a batch

        Outputs:
        1) vec -> an array containing a conditional vector for each data point to be generated 
        """

        if self.n_col == 0:
            return None
        
        batch = batch

        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        
        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):   
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec

class Sampler(object):
    
    """
    This class is used to sample the transformed real data according to the conditional vector 

    Variables:
    1) data -> real transformed input data
    2) model -> stores the index values of data records corresponding to any given selected categories for all columns
    3) n -> size of the input data

    Methods:
    1) __init__() -> initiates the sampler object and stores class variables 
    2) sample() -> takes as input the number of rows to be sampled (n), chosen column (col)
                   and category within the column (opt) to sample real records accordingly
    """

    def __init__(self, data, output_info):
        
        super(Sampler, self).__init__()
        
        self.data = data
        self.model = []
        self.n = len(data)
        
        # counter to iterate through columns
        st = 0
        # iterating through column information
        for item in output_info:
            # ignoring numeric columns
            if item[1] == 'tanh':
                st += item[0]
                continue
            # storing indices of data records for all categories within one-hot-encoded representations
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                # iterating through each category within a one-hot-encoding
                for j in range(item[0]):
                    # storing the relevant indices of data records for the given categories
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
                
    def sample(self, n, col, opt):
        
        # if there are no one-hot-encoded representations, we may ignore sampling using a conditional vector
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        
        # used to store relevant indices of data records based on selected category within a chosen one-hot-encoding
        idx = []
        
        # sampling a data record index randomly from all possible indices that meet the given criteria of the chosen category and one-hot-encoding
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        
        return self.data[idx], idx


class MDGANServer():
    """
    This is the class that encapsulates the functions that need to be run on the server side
    This is the main driver of the training procedure.
    """

    def __init__(self, client_rrefs, epochs = 300, use_cuda = False, batch_size = 500, n_critic = 5, **kwargs):
        # super(MDGANServer, self).__init__(**kwargs)
        print("number of epochs in initialization: ", epochs)
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cpu':
            self.use_cuda = True
        self.n_critic = n_critic
        self.batch_size = batch_size
  
        # keep a reference to the client
        self.client_rrefs = []
        for client_rref in client_rrefs:
            self.client_rrefs.append(client_rref)

        self.embedding_dim=128
        self.generator_dim=(256, 256)
        self.discriminator_dim=(256, 256)
        self.data_dim = 256 # this is the fixed output dim of generator


        # block to init generator and discriminator

        self.d_input_dim_list = [] # record the coded data dim of each client, it is also the input dim for Partial Discriminator from each client.
        for client_rref in self.client_rrefs:
            self.d_input_dim_list.append(client_rref.remote().get_data_dim().to_here())
            
        self.d_output_dim_list = []
        for var in self.d_input_dim_list:
            self.d_output_dim_list.append(int(var/self.data_dim))
        
        for client_rref, dim in zip(self.client_rrefs, self.d_output_dim_list):
            client_rref.rpc_sync().init_local(dim)
        
        # check if the sum of self.d_output_dim_list == self.data_dim
        # if not, adjust last one to make it even
        if  np.sum(self.d_output_dim_list) != self.data_dim:
            self.d_output_dim_list[-1] = int(self.data_dim - np.sum(self.d_output_dim_list[0:-1]))

        


        self.generator = Generator(
            self.embedding_dim,
            self.generator_dim,
            self.data_dim,
            self.d_output_dim_list
        ).to(self.device)


        self.discriminator = Discriminator(
            self.data_dim,
            self.discriminator_dim,
            self.d_input_dim_list,
            self.d_output_dim_list
        ).to(self.device)

        self.param_rrefs_list_g = []
        self.param_rrefs_list_g.append(param_rrefs(self.generator))
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_g.append(client_rref.rpc_sync().register_local_g_in_server().to_here())

        self.G_opt = DistributedOptimizer(
           optim.Adam, self.param_rrefs_list_g, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )

        self.param_rrefs_list_d = []
        self.param_rrefs_list_d.append(param_rrefs(self.discriminator))
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_d.append(client_rref.rpc_sync().register_local_d_in_server().to_here())

        self.D_opt = DistributedOptimizer(
           optim.Adam, self.param_rrefs_list_d, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )


        self.columns_name = []
        for client_rref in self.client_rrefs:
            if self.columns_name == []:
                self.columns_name = client_rref.rpc_sync().get_columns_name().to_here()
            else:              
                self.columns_name =+ client_rref.rpc_sync().get_columns_name().to_here()

    def init_discriminator(self):
        '''
        This function should be called ONLY AFTER the init of clients.
        '''
        self.discriminator = Discriminator(
            self.data_dim,
            self.discriminator_dim,
            self.d_dim_list
        ).to(self.device)

    def init_generator_optimizer(self):
        self.param_rrefs_list_g = []
        self.param_rrefs_list_g.append(param_rrefs(self.generator))
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_g.append(client_rref.rpc_sync().register_local_g_in_server().to_here())

        self.G_opt = DistributedOptimizer(
           optim.Adam, self.param_rrefs_list_g, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )


    def init_discriminator_optimizer(self):
        self.param_rrefs_list_d = []
        self.param_rrefs_list_d.append(param_rrefs(self.discriminator))
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_d.append(client_rref.rpc_sync().register_local_d_in_server().to_here())

        self.D_opt = DistributedOptimizer(
           optim.Adam, self.param_rrefs_list_d, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )


    def get_discriminator_weights(self):
        print("call in get_discriminator_weights: ")
        if next(self.discriminator.parameters()).is_cuda:
            return self.discriminator.cpu().state_dict()
        else:
            return self.discriminator.state_dict()

    def set_discriminator_weights(self, state_dict):
        print("call in set_discriminator_weights: ", self.device)
        self.discriminator.load_state_dict(state_dict)
        if self.device.type != 'cpu':
            print("set discriminator on cuda")
            self.discriminator.to(self.device)

    def reset_on_cuda(self):
        if self.device.type != 'cpu':
            self.discriminator.to(self.device)

    def sample_generation(self):
        z = Variable(torch.randn(self.batch_size, *self.latent_shape))
        if self.use_cuda:
            z = z.cuda()
        return self.generator(z)


    # def save_gif(self):
    #     # grid = make_grid(self.G(self._fixed_z).cpu().data, normalize=True)
    #     # grid = np.transpose(grid.numpy(), (1, 2, 0))
    #     # self.images.append(grid)
    #     imageio.mimsave('{}.gif'.format('mnist'), self.images)

    def gradient_penalty(self, data, generated_data, gamma=10):
        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1)
        epsilon = epsilon.expand_as(data)


        if self.use_cuda:
            epsilon = epsilon.cuda()

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = Variable(interpolation, requires_grad=True)

        if self.use_cuda:
            interpolation = interpolation.cuda()
        # print("shape of data", data.shape, generated_data.shape, interpolation.shape)
        interpolation_logits = self.discriminator(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        # here add an epsilon to avoid sqrt error for number close to 0
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12) 
        return gamma * ((gradients_norm - 1) ** 2).mean()


    def fit(self):

        """
        E: the interval epochs to swap models
        """
        training_data_length = self.client_rrefs[0].rpc_sync().get_data_length().to_here()
        steps_per_epoch = max(training_data_length // self.batch_size, 1)
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        for i in range(self.epochs):
            for id_ in range(steps_per_epoch):

                for n in range(self.n_critic):
                    with dist.autograd.context() as D_context:
                        fakez = torch.normal(mean=mean, std=std)
                        fake_list = self.generator(fakez)
                        intermediate_list_fake = []
                        intermediate_list_real = []
                        for client_rref, fake_ in zip(self.client_rrefs, fake_list):
                            inter_fake, inter_real = client_rref.remote().get_intermediate_result_both(fake_).to_here()
                            intermediate_list_fake.append(inter_fake)
                            intermediate_list_real.append(inter_real)

                        for i in range(len(intermediate_list_fake)):
                            intermediate_list_fake[i] = intermediate_list_fake[i].wait()
                            intermediate_list_real[i] = intermediate_list_real[i].wait()

                        fake_input = torch.cat(intermediate_list_fake, dim = 1)
                        real_input = torch.cat(intermediate_list_real, dim = 1)
                        y_fake = self.discriminator(fake_input)
                        y_real = self.discriminator(real_input)

                        pen = self.discriminator.calc_gradient_penalty(
                            real_input, fake_input, self.device)
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                        dist.autograd.backward(D_context, [pen+loss_d])
                        self.D_opt.step(D_context)

                with dist.autograd.context() as G_context:  
                    fakez = torch.normal(mean=mean, std=std)
                    fake_list = self.generator(fakez)
                    intermediate_list_fake = []
                    for client_rref, fake_ in zip(self.client_rrefs, fake_list):
                        inter_fake, inter_real = client_rref.remote().get_intermediate_result_for_generator(fake_).to_here()
                        intermediate_list_fake.append(inter_fake)
                    for i in range(len(intermediate_list_fake)):
                        intermediate_list_fake[i] = intermediate_list_fake[i].wait()
                    fake_input = torch.cat(intermediate_list_fake, dim = 1)
                    y_fake = self.discriminator(fake_input)
                    dist.autograd.backward(G_context, [-torch.mean(y_fake)])
                    self.G_opt.step(D_context)

    def sample(self, sample_size = 5000):
        mean = torch.zeros(sample_size, self.embedding_dim, device=self.device)
        std = mean + 1       
        fakez = torch.normal(mean=mean, std=std)
        fake_list = self.generator(fakez)
        intermediate_list_fake = []
        for client_rref, fake_ in zip(self.client_rrefs, fake_list):
            inter_fake, inter_real = client_rref.remote().sample(fake_).to_here()
            intermediate_list_fake.append(inter_fake)
        for i in range(len(intermediate_list_fake)):
            intermediate_list_fake[i] = intermediate_list_fake[i].wait()
        fake_input = torch.cat(intermediate_list_fake, dim = 1)

        pd.DataFrame(fake_input.numpy(), columns = self.columns_name).to_csv("sample.csv", index=False)

        return fake_input                    

            



class MDGANClient():
    """
    This is the class that encapsulates the functions that need to be run on the client side
    Despite the name, this source code only needs to reside on the server, the real MDGANClient
    will be initialized with the code in client side via RPC() call.
    """

    def __init__(self, dataset, epochs, use_cuda, batch_size, **kwargs):
        print("number of epochs in initialization: ", epochs)
        print("initalized dataset: ", dataset)
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if self.device.type != 'cpu':
            self.use_cuda = True

        # self.raw_df = pd.read_csv("Adult.csv")
        # self.column_number = self.raw_df.shape[1]
        # self.columns_name = list(self.raw_df.columns)
        # self.test_ratio = 0
        # self.categorical_columns = ['education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
        # self.log_columns = []
        # self.mixed_columns= {}
        # self.integer_columns = ['age', 'fnlwgt']
        # self.problem_type= {None: None}



        self.split_threshold = 7
        self.raw_df = pd.read_csv("Loan.csv").iloc[:,self.split_threshold:]
        print("raw_df shape: ", self.raw_df.shape)
        self.column_number = self.raw_df.shape[1]
        self.columns_name = list(self.raw_df.columns)
        self.test_ratio = 0
        whole_categorical = ["ZIP Code","Family","Education","PersonalLoan","Securities Account","CD Account","Online","CreditCard"]
        self.categorical_columns = []
        for col in self.raw_df.columns:
            if col in whole_categorical:
                self.categorical_columns.append(col)

        self.log_columns = []

        whole_mixed_columns = {"Mortgage":[0.0]}
        self.mixed_columns = {}
        for col in self.raw_df.columns:
            if col in whole_mixed_columns.keys():
                self.mixed_columns[col] = whole_mixed_columns[col]

        whole_integer_columns = ["Age", "Experience", "Income","Mortgage"]
        self.integer_columns = []

        
        for col in self.raw_df.columns:
            if col in whole_integer_columns:
                self.integer_columns.append(col)
   
        print("categorical_columns: ", self.categorical_columns)
        print("log_columns: ", self.log_columns)
        print("mixed_columns: ", self.mixed_columns)
        print("integer_columns: ", self.integer_columns)

        self.problem_type= {None: None}


        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        print(self.data_prep.df.head())
        self.transformer = DataTransformer(train_data=self.data_prep.df, categorical_list=self.data_prep.column_types["categorical"], mixed_dict=self.data_prep.column_types["mixed"])
        self.transformer.fit() 
        
        self.train_data = self.transformer.transform(self.data_prep.df.values)
        self.data_dim = self.transformer.output_dim

        # initializing the sampler object to execute training-by-sampling 
        self.data_sampler = Sampler(self.train_data, self.transformer.output_info)
        # initializing the condvec object to sample conditional vectors during training
        self.cond_generator = Condvec(self.train_data, self.transformer.output_info)

        # self.local_d = Linear(self.data_dim, self.data_dim)
        self.locator = 0
        

        print("train data shape: ", self.train_data.shape)
        print("output info: ", self.transformer.output_info)
        print("data_dim: ", self.transformer.output_dim)
        print("cond_generator length: ", self.cond_generator.n_opt)

        # construct local dataloader
        self.data_loader = DataLoader(self.train_data, self.batch_size, shuffle=False)


    def shuffle_training_data(self):
        
        order = shuffle(range(self.train_data.shape[0]), random_state=42)
        # print("shuffle order: ", order[0:10],self.train_data.shape, type(self.train_data))
        index = order.index(self.locator)
        # print("index: ", index)
        self.train_data = self.train_data[order]
        self.locator = index
        # print("train data[locator]", self.train_data[self.locator])
        # print("transformer output info: ", self.transformer.output_info)
        # initializing the sampler object to execute training-by-sampling 
        self.data_sampler = Sampler(self.train_data, self.transformer.output_info)
        # initializing the condvec object to sample conditional vectors during training
        self.cond_generator = Condvec(self.train_data, self.transformer.output_info)


    
    def get_intermediate_result_both(self, generator_input):
        # for fake data     
        intermediate_output_fake = self.local_d(apply_activate(self.local_g(generator_input.to(self.device)), self.transformer.output_info))

        # for real data
        iterloader = iter(self.data_loader)
        try:
            data = next(iterloader)
        except StopIteration:
            iterloader = iter(self.data_loader)
            data = next(iterloader)

        intermediate_output_real = self.local_d(data.float())

        if self.use_cuda:
            return intermediate_output_fake.cpu(), intermediate_output_real.cpu()
        else:
            return intermediate_output_fake, intermediate_output_real


    def get_whole_output(self, idx=None):
        if idx is not None:
            intermediate_output_real = self.local_d(torch.from_numpy(self.train_data[idx]).float().to(self.device))
        else:
            intermediate_output_real = self.local_d(torch.from_numpy(self.train_data).float().to(self.device))
        if self.use_cuda:
            return intermediate_output_real.cpu()
        else:
            return intermediate_output_real

    def get_convec_corresponding_output(self):

        # sampling real data according to the conditional vectors and shuffling it before feeding to discriminator to isolate conditional loss on generator    
        perm = np.arange(self.batch_size)
        np.random.shuffle(perm)
        real, idx = self.data_sampler.sample(self.batch_size, self.col[perm], self.opt[perm])
        real = torch.from_numpy(real.astype('float32')).to(self.device)

        intermediate_output_real = self.local_d(real.float())

        # print("in chosen client: ", self.c.shape, real.shape, idx[0:10])

        if self.use_cuda:
            return intermediate_output_real.cpu(), idx, self.c[perm].cpu()
        else:
            return intermediate_output_real, idx, self.c[perm]


    def get_intermediate_result_for_generator(self, generator_input, isCalculated_generator_loss = False):
        faket = self.local_g(generator_input.to(self.device))
        synthetic_data = apply_activate(faket, self.transformer.output_info)

        cross_entropy = -1
        if isCalculated_generator_loss:
            cross_entropy = cond_loss(faket, self.transformer.output_info, self.c, self.m)
            # print("cross_entropy: ", cross_entropy)

        intermediate_output_fake = self.local_d(synthetic_data)

        # for name, param in self.local_g.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        # for name, param in self.local_d.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        if self.use_cuda:
            # print("send back cpu tensor")
            if isCalculated_generator_loss:
                return intermediate_output_fake.cpu(), cross_entropy.cpu()
            else:
                return intermediate_output_fake.cpu(), cross_entropy
        else:
            return intermediate_output_fake, cross_entropy


    def get_condvec(self, sample_size):
        condvec = self.cond_generator.sample_train(sample_size)
        self.c, self.m, self.col, self.opt = condvec
        self.c = torch.from_numpy(self.c).to(self.device)
        self.m = torch.from_numpy(self.m).to(self.device)       

        if self.use_cuda:
            return self.c.cpu()
        else:
            return self.c


    def sample(self, generator_input):
        generator_input = generator_input.to(self.device)
        output_g = self.local_g(generator_input).detach().cpu()
        activation = apply_activate(output_g, self.transformer.output_info)
        inverse_transform = self.transformer.inverse_transform(activation)
        return self.data_prep.inverse_prep(inverse_transform)
        
    def init_local(self, dim):
        print("dim and self.data_dim", dim, self.data_dim)

        self.local_g = Linear(dim, self.data_dim).to(self.device)
        self.local_d = Linear(self.data_dim, dim).to(self.device)

        if self.use_cuda:
            self.local_g = self.local_g.cuda()
            self.local_d = self.local_d.cuda()

    def get_column_number(self):
        return self.column_number

    def get_condvec_length(self):
        return self.cond_generator.n_opt

    def get_columns_name(self):
        return self.columns_name

    def get_data_length(self):
        return len(self.train_data)

    def get_data_dim(self):
        return self.data_dim
        
    def register_local_d_in_server(self):
        return param_rrefs(self.local_d)

    def register_local_g_in_server(self):
        return param_rrefs(self.local_g)

    def send_client_refs(self):
        """Send a reference to the discriminator (for future RPC calls) and the conditioner, transformer and steps/epoch"""
        return rpc.RRef(self.discriminator)

    def register_G(self, G_rref):
        """Receive a reference to the generator (for future RPC calls)"""
        self.G_rref = G_rref



   




def run(rank, world_size, ip, port, dataset, epochs, use_cuda, batch_size, n_critic):
    # set environment information
    os.environ["MASTER_ADDR"] = ip
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    print("number of epochs before initialization: ", epochs)
    print("world size: ", world_size, f"tcp://{ip}:{port}")
    if rank == 0:  # this is run only on the server side
        print("server")
        rpc.init_rpc(
            "server",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.PROCESS_GROUP,
            rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                num_send_recv_threads=8, rpc_timeout=120, init_method=f"tcp://{ip}:{port}"
            ),
        )
        print("after init_rpc")
        clients = []
        for worker in range(world_size-1):
            clients.append(rpc.remote("client"+str(worker+1), MDGANClient, kwargs=dict(dataset=dataset, epochs = epochs, use_cuda = use_cuda, batch_size=batch_size)))
            print("register remote client"+str(worker+1), clients[0])

        synthesizer = MDGANServer(clients, epochs, use_cuda, batch_size, n_critic)
        synthesizer.fit()

    elif rank != 0:
        rpc.init_rpc(
            "client"+str(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.PROCESS_GROUP,
            rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                num_send_recv_threads=8, rpc_timeout=120, init_method=f"tcp://{ip}:{port}"
            ),
        )
        print("client"+str(rank)+" is joining")

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rank", type=int, default=0)
    parser.add_argument("-ip", type=str, default="127.0.0.1")
    parser.add_argument("-port", type=int, default=7788)
    parser.add_argument(
        "-dataset", type=str, default="mnist"
    )
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-world_size", type=int, default=2)
    parser.add_argument('-use_cuda',  type=bool, default=False)
    parser.add_argument("-batch_size", type=int, default=500)
    parser.add_argument("-n_critic", type=int, default=1)
    args = parser.parse_args()

    if args.rank is not None:
        # run with a specified rank (need to start up another process with the opposite rank elsewhere)
        run(
            rank=args.rank,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            dataset=args.dataset,
            epochs=args.epochs,
            use_cuda=args.use_cuda,
            batch_size=args.batch_size,
            n_critic=args.n_critic
        )
