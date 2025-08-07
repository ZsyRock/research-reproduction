import argparse
import torch
import pandas as pd
from torch.nn import Linear
import torch.distributed.rpc as rpc
from torch.autograd import Variable
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import autograd

import numpy as np
import os
import random
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential

import warnings
warnings.filterwarnings("ignore")

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
        # for item in list(generator_dim):
        #     seq += [Residual(dim, item)]
        #     dim += item
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

class GeneratorSplitStepOne(Module):
    """Generator: comparing two original Generator, split the two blocks: one in server and one in client, it will move to client side
    """

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

class LocalGenerator(Module):
    """local generator with one residual block"""
    def __init__(self, input_dim, local_generator_dim, data_dim):
        '''
        data_dim is the output dimension, same as the encoded data dimension
        '''
        super(LocalGenerator, self).__init__()
        seq = []
        for item in list(local_generator_dim):
            seq += [Residual(input_dim, item)]
            input_dim += item
        seq.append(Linear(input_dim, data_dim))
        self.seq = Sequential(*seq)        

    def forward(self, input):
        """Apply the LocalGenerator to the `input`."""
        return self.seq(input)

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
        self.generator_dim=[] #(256, 256)
        self.discriminator_dim=(256, 256)
        self.generator_division = 256

        # get global conditional vector length
        self.cond_vec_length = 0
        self.cond_vec_length_list = []
        for client_rref in self.client_rrefs:
            length = client_rref.rpc_sync().get_condvec_length()
            print("cond_vec_length: ", length)
            self.cond_vec_length_list.append(length)
            self.cond_vec_length += length

        self.data_dim = self.embedding_dim +  self.cond_vec_length # this is the fixed output dim of generator


        # block to init generator and discriminator

        self.d_input_dim_list = [] # record the coded data dim of each client, it is also the input dim for Partial Discriminator from each client.
        for client_rref in self.client_rrefs:
            self.d_input_dim_list.append(client_rref.remote().get_data_dim().to_here())
        print("d_input_dim_list: ", self.d_input_dim_list)   
        self.d_output_dim_list = []
        self.generator_division_list = []
        for var in self.d_input_dim_list:
            self.d_output_dim_list.append(int(var*self.data_dim/np.sum(self.d_input_dim_list)))
            self.generator_division_list.append(int(var*self.generator_division/np.sum(self.d_input_dim_list)))

           
        # record number of columns in each client.
        self.columns_number = []
        for client_rref in self.client_rrefs:
            self.columns_number.append(client_rref.remote().get_column_number().to_here())
        # ratio will be used as the probability to select client for constructing conditional vector
        self.column_number_ratio = []
        for var in self.columns_number:
            self.column_number_ratio.append(np.round(var/np.sum(self.columns_number), 2))
        if np.sum(self.column_number_ratio) != 1:
            self.column_number_ratio[-1] = int(1 - np.sum(self.column_number_ratio[0:-1]))




        # check if the sum of self.d_output_dim_list == self.data_dim
        # if not, adjust last one to make it even
        if  np.sum(self.d_output_dim_list) != self.data_dim:
            self.d_output_dim_list[-1] = int(self.data_dim - np.sum(self.d_output_dim_list[0:-1]))
        if  np.sum(self.generator_division_list) != self.generator_dim:
            self.generator_division_list[-1] = int(self.generator_dim - np.sum(self.generator_division_list[0:-1]))
        print("d_output_dim_list: ", self.d_output_dim_list) 
        print("generator_division_list: ", self.generator_division_list) 

        # initialize intermediate g and d in client side
        for client_rref, dim, division in zip(self.client_rrefs, self.d_output_dim_list, self.generator_division_list):
            client_rref.rpc_sync().init_local(dim, division)        

        # this local_d is used to let conditional vector to go through, because other real data already go through d1 d2, their output should not directly connect to conditional vector.
        self.local_d = Linear(self.cond_vec_length, self.cond_vec_length)
        if self.use_cuda:
            self.local_d = self.local_d.cuda()

        self.generator = Generator(
            self.embedding_dim+self.cond_vec_length,
            self.generator_dim,
            self.data_dim,
            self.d_output_dim_list
        ).to(self.device)
        print("generator model:", self.generator)
        

        self.discriminator = Discriminator(
            np.sum(self.d_input_dim_list)+self.cond_vec_length,
            self.discriminator_dim
        ).to(self.device)
        print("discriminator model:", self.discriminator)
        print("discriminator input dim: ", self.data_dim+self.cond_vec_length)
        self.param_rrefs_list_g = param_rrefs(self.generator)
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_g += client_rref.rpc_sync().register_local_g_in_server()

        self.G_opt = DistributedOptimizer(
           optim.Adam, self.param_rrefs_list_g, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )

        self.param_rrefs_list_d = param_rrefs(self.discriminator)
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_d += client_rref.rpc_sync().register_local_d_in_server()

        self.D_opt = DistributedOptimizer(
           optim.Adam, self.param_rrefs_list_d, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )


        self.columns_name = []
        for client_rref in self.client_rrefs:
            if self.columns_name == []:
                self.columns_name = client_rref.rpc_sync().get_columns_name()
            else:              
                self.columns_name += client_rref.rpc_sync().get_columns_name()

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
            self.param_rrefs_list_g.append(client_rref.rpc_sync().register_local_g_in_server())

        self.G_opt = DistributedOptimizer(
           optim.Adam, self.param_rrefs_list_g, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )


    def init_discriminator_optimizer(self):
        self.param_rrefs_list_d = []
        self.param_rrefs_list_d.append(param_rrefs(self.discriminator))
        self.param_rrefs_list_d.append(param_rrefs(self.local_d))
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_d.append(client_rref.rpc_sync().register_local_d_in_server())

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

    def construction_condvec(self, sample_size):

        # chosen_client contributes the conditional vector to global training
        self.chosen_client = random.choices(range(len(self.client_rrefs)), weights=self.column_number_ratio, k=1)[0]
        cond_vec_list = []
        for idx, client_rref in enumerate(self.client_rrefs):
            if idx != self.chosen_client:
                cond_vec = client_rref.remote().get_condvec(sample_size).to_here()
                cond_vec_list.append(np.zeros_like(cond_vec))
            else:
                cond_vec_list.append(client_rref.remote().get_condvec(sample_size).to_here())

        return np.concatenate(cond_vec_list, axis=1)

    def reconstruct_condvec(self, partial_condvec, sample_size):

        cond_vec_list = []
        for idx, client_rref in enumerate(self.client_rrefs):
            if idx != self.chosen_client:
                cond_vec = client_rref.remote().get_condvec(sample_size).to_here()
                cond_vec_list.append(np.zeros_like(cond_vec))
            else:
                cond_vec_list.append(partial_condvec)

        return np.concatenate(cond_vec_list, axis=1)        


    def fit(self):

        """
        E: the interval epochs to swap models
        """
        training_data_length = self.client_rrefs[0].rpc_sync().get_data_length()
        steps_per_epoch = max(training_data_length // self.batch_size, 1)
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        for epoch in range(self.epochs):

            print("epoch: ", epoch)
            for client_rref in self.client_rrefs:
                    client_rref.rpc_sync().shuffle_training_data() 

            for id_ in range(steps_per_epoch):
                for n in range(self.n_critic):
                    with dist.autograd.context() as D_context:
                        noisez = torch.normal(mean=mean, std=std)
                        # construct conditional vector
                        condvec = self.construction_condvec(self.batch_size)
                        # print("chosen client for fake: ", self.chosen_client)
                        fake_list = self.generator(torch.cat([noisez, torch.from_numpy(condvec).to(self.device)], dim=1))
                        intermediate_list_fake = []
                        intermediate_list_real = []
                        future_object_list = []

                        # get output from generator
                        for client_rref, fake_ in zip(self.client_rrefs, fake_list):
                            if self.use_cuda:
                                future_object_list.append(client_rref.rpc_async().get_intermediate_result_for_generator(fake_.cpu(), False))
                            else:
                                future_object_list.append(client_rref.rpc_async().get_intermediate_result_for_generator(fake_, False))

                        for i in range(len(future_object_list)):
                            fake_inter, _ = future_object_list[i].wait()
                            # print("D training fake intermediate value shape at client "+str(i)+": ", fake_inter.shape)
                            intermediate_list_fake.append(fake_inter)

                        # get output from discriminator
                        future_object_list = []
                        for idx_, client_rref in enumerate(self.client_rrefs): 
                            if idx_ == self.chosen_client:
                                future_object_list.append(client_rref.rpc_async().get_convec_corresponding_output())
                            else:
                                future_object_list.append(client_rref.rpc_async().get_whole_output())
                                
                        # print("chosen client for real: ", self.chosen_client)
                        real_inter_chosen, idx, c_perm = future_object_list[self.chosen_client].wait()

                        for i in range(len(future_object_list)):
                            if i == self.chosen_client:
                                intermediate_list_real.append(real_inter_chosen)
                                # print("D training real intermediate value shape at client "+str(i)+": ", real_inter_chosen.shape)
                            else:
                                real_inter = future_object_list[i].wait()
                                intermediate_list_real.append(real_inter[idx])
                                # print("D training real intermediate value shape at client "+str(i)+": ", real_inter[idx].shape)
                        # ZZ: it's important and CORRECT to make conditional vector go through local_d, this is the local filter layer in SERVER side!!!
                        intermediate_list_fake.append(self.local_d(torch.from_numpy(condvec).to(self.device)).cpu())
                        intermediate_list_real.append(self.local_d(torch.from_numpy(self.reconstruct_condvec(c_perm, self.batch_size)).to(self.device)).cpu())
                        # print("c_perm dimision: ", c_perm.shape, self.reconstruct_condvec(c_perm).shape)
                        fake_input = torch.cat(intermediate_list_fake, dim = 1).to(self.device)
                        real_input = torch.cat(intermediate_list_real, dim = 1).to(self.device)
                        # print("fake_input_shape: ", fake_input.shape, real_input.shape)
                        y_fake = self.discriminator(fake_input)
                        y_real = self.discriminator(real_input)
                        
                        pen = self.discriminator.calc_gradient_penalty(
                            real_input, fake_input, self.device)
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                        
                        dist.autograd.backward(D_context, [pen+loss_d])
                        
                        self.D_opt.step(D_context)
                        

                with dist.autograd.context() as G_context:  
                    # print("G training")
                    noisez = torch.normal(mean=mean, std=std)
                    condvec = self.construction_condvec(self.batch_size)
                    fake_list = self.generator(torch.cat([noisez, torch.from_numpy(condvec).to(self.device)], dim=1))
                    intermediate_list_fake = []

                    for client_rref, fake_ in zip(self.client_rrefs, fake_list):
                        if self.use_cuda:
                            fake_inter = client_rref.rpc_async().get_intermediate_result_for_generator(fake_.cpu(), True)
                        else:
                            fake_inter = client_rref.rpc_async().get_intermediate_result_for_generator(fake_, True)
                        intermediate_list_fake.append(fake_inter)                        
                    for i in range(len(intermediate_list_fake)):
                        if i == self.chosen_client:
                            intermediate_list_fake[i], cross_entropy = intermediate_list_fake[i].wait()
                        else:
                            intermediate_list_fake[i], _ = intermediate_list_fake[i].wait()

                    # input to discriminator also needs to add conditional vector
                    intermediate_list_fake.append(self.local_d(torch.from_numpy(condvec).to(self.device)).cpu())
                    fake_input = torch.cat(intermediate_list_fake, dim = 1).to(self.device)
                    y_fake = self.discriminator(fake_input)                  
                    dist.autograd.backward(G_context, [-torch.mean(y_fake), cross_entropy])
                    self.G_opt.step(G_context)
            print("generator loss: ", cross_entropy)
            if (epoch+1) % 10 == 0:
                print("sampling data at epoch: ",epoch)
                self.sample(epoch, 40000)
        
    def sample(self, epoch = 0, sample_size = 5000):
        mean = torch.zeros(sample_size, self.embedding_dim, device=self.device)
        std = mean + 1       
        noisez = torch.normal(mean=mean, std=std)
        condvec = self.construction_condvec(sample_size)
        fake_list = self.generator(torch.cat([noisez, torch.from_numpy(condvec).to(self.device)], dim=1))
        intermediate_list_fake = []
        for client_rref, fake_ in zip(self.client_rrefs, fake_list):
            if self.use_cuda:
                inter_fake = client_rref.remote().sample(fake_.cpu()).to_here()
            else:
                inter_fake = client_rref.remote().sample(fake_).to_here()
            # print(inter_fake)
            intermediate_list_fake.append(inter_fake)
        # for i in range(len(intermediate_list_fake)):
        #     intermediate_list_fake[i] = intermediate_list_fake[i].wait()
        
        fake_input = pd.concat(intermediate_list_fake, axis = 1)
        fake_input.to_csv("generation/sample_{}.csv".format(epoch), index=False)

        # pd.DataFrame(fake_input.values, columns = self.columns_name).to_csv("sample.csv", index=False)

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
        self.use_cuda = False
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if self.device.type != 'cpu':
            self.use_cuda = True

        self.raw_df = pd.read_csv("Adult.csv")
        self.column_number = self.raw_df.shape[1]
        self.columns_name = list(self.raw_df.columns)
        self.test_ratio = 0,
        self.categorical_columns = [ 'workclass', 'native-country', 'income']
        self.log_columns = []
        self.mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]}
        self.integer_columns = ['capital-gain', 'capital-loss','hours-per-week']
        self.problem_type= {None: None}


        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        print(self.data_prep.df.head())
        print(self.data_prep.column_types["categorical"])
        self.transformer = DataTransformer(train_data=self.data_prep.df, categorical_list=self.data_prep.column_types["categorical"], mixed_dict=self.data_prep.column_types["mixed"])
        self.transformer.fit() 
        
        self.train_data = self.transformer.transform(self.data_prep.df.values)
        self.data_dim = self.transformer.output_dim

        # initializing the sampler object to execute training-by-sampling 
        self.data_sampler = Sampler(self.train_data, self.transformer.output_info)
        # initializing the condvec object to sample conditional vectors during training
        self.cond_generator = Condvec(self.train_data, self.transformer.output_info)


        self.local_d = Linear(self.data_dim, self.data_dim)

        print("train data shape: ", self.train_data.shape)
        print("output info: ", self.transformer.output_info)
        print("data_dim: ", self.transformer.output_dim)
        


        # construct local dataloader
        self.data_loader = DataLoader(self.train_data, self.batch_size, shuffle=False)

    def shuffle_training_data(self):
        order = shuffle(range(self.train_data.shape[0]), random_state=42)
        self.train_data = self.train_data[order]
        # initializing the sampler object to execute training-by-sampling 
        self.data_sampler = Sampler(self.train_data, self.transformer.output_info)
        # initializing the condvec object to sample conditional vectors during training
        self.cond_generator = Condvec(self.train_data, self.transformer.output_info)


    def get_intermediate_result_both(self, generator_input):
        # for fake data     
        intermediate_output_fake = self.local_d(apply_activate(self.local_g(generator_input), self.transformer.output_info))

        # for real data
        iterloader = iter(self.data_loader)
        try:
            data = next(iterloader)
        except StopIteration:
            iterloader = iter(self.data_loader)
            data = next(iterloader)

        intermediate_output_real = self.local_d(data.float())

        return intermediate_output_fake, intermediate_output_real


    def get_whole_output(self):

        intermediate_output_real = self.local_d(self.train_data.float())
        return intermediate_output_real

    def get_convec_corresponding_output(self):

        # sampling real data according to the conditional vectors and shuffling it before feeding to discriminator to isolate conditional loss on generator    
        perm = np.arange(self.batch_size)
        np.random.shuffle(perm)
        real, idx = self.data_sampler.sample(self.batch_size, self.col[perm], self.opt[perm])
        real = torch.from_numpy(real.astype('float32')).to(self.device)

        intermediate_output_real = self.local_d(real.float())
        return intermediate_output_real, idx




    def get_intermediate_result_for_generator(self, generator_input, isCalculated_generator_loss = False):
        faket = self.local_g(generator_input.to(self.device))
        synthetic_data = apply_activate(faket, self.transformer.output_info)

        cross_entropy = -1
        if isCalculated_generator_loss:
            cross_entropy = cond_loss(faket, self.transformer.output_info, self.c, self.m)

        intermediate_output_fake = self.local_d(synthetic_data)

        # for name, param in self.local_g.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        # for name, param in self.local_d.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        if self.use_cuda:
            # print("send back cpu tensor")
            return intermediate_output_fake.cpu(), cross_entropy
        else:
            return intermediate_output_fake, cross_entropy


    def get_condvec(self, sample_size):
        condvec = self.cond_generator.sample_train(sample_size)
        self.c, self.m, self.col, self.opt = condvec
        return self.c


    def sample(self, generator_input):
        return self.data_prep.inverse_prep(self.transformer.inverse_transform(apply_activate(self.local_g(generator_input), self.transformer.output_info).detach().numpy()))
        

    def init_local(self, dim):
        #print("dim and self.data_dim", dim, self.data_dim)
        self.local_g = LocalGenerator(input_dim = dim, local_generator_dim = (256), data_dim = self.data_dim)
        #print("local generator model:", self.local_g)
        # Linear(dim, self.data_dim)
        self.local_d = Linear(self.data_dim, dim)
        if self.use_cuda:
            self.local_g = self.local_g.cuda()
            self.local_d = self.local_d.cuda()

    def get_column_number(self):
        return self.column_number

    def get_condvec_length(self):
         self.cond_generator.n_opt

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
                num_send_recv_threads=8, rpc_timeout=10000, init_method=f"tcp://{ip}:{port}"
            ),
        )
        print("after init_rpc")
        clients = []
        for worker in range(world_size-1):
            clients.append(rpc.remote("client"+str(worker+1), MDGANClient, kwargs=dict(dataset=dataset, epochs = epochs, use_cuda = use_cuda, batch_size=batch_size)))
            print("register remote client"+str(worker+1), clients[0])
        for worker in range(world_size-1):
            clients[worker].to_here()
           
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
