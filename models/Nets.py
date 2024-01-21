#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F
from random import shuffle
from models.model_partition_tool import *
from models.Fed import FedAvg

class MLP(nn.Module):
    def __init__(self, dim_in, model_size, dim_out, partition_num=1):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.partition_num = partition_num
        self.model_size = model_size
        self.temp_hidden_layer_index = [i for i in range(self.model_size)]
        self.fc1 = nn.Linear(self.dim_in, self.model_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(self.model_size, self.dim_out)
        # When the server partition the model, we need to mark
        # the partition details of sub-models
        if self.partition_num != 1:
            self.partition_dim = create_list(self.model_size, self.partition_num)
            self.hidden_layer_index_log = []
            self.fc1_weight_partition = []
            self.fc2_weight_partition = []

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # The server will call this function when it finishes the model synchronization.
    # Partition weights, bias is neglected, the effect is limitted
    def partition_to_list(self):
        print("Repartition parameters!")
        shuffle(self.temp_hidden_layer_index) # self.temp_hidden_layer_index = [0,1,...,self.partition_dim-1]
        self.hidden_layer_index_log.clear()
        for i in range(self.partition_num):
            current_indexes = torch.tensor(
                self.temp_hidden_layer_index[i * self.partition_dim[0]:i * self.partition_dim[0] + self.partition_dim[i]])
            self.hidden_layer_index_log.append(current_indexes)
        self.fc1_weight_partition.clear()
        self.fc2_weight_partition.clear()
        # record the weight for each cell
        self.fc1_weight_partition = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.hidden_layer_index_log)
        # return a list containing first-layer weight for each cell
        self.fc2_weight_partition = partition_FC_layer_by_input_dim_1(
            self.fc2.weight, self.hidden_layer_index_log)
        # return a list containing final-layer weight for each cell

    def flush(self):
        # update the model based on the collected parameters.
        # Here we have to get around pytorch variable by use variable.data,
        # since leaf variable disallowing in-place operation
        # Note: only global server will call this method to synchronize all the models from edge servers
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition,
                                            self.hidden_layer_index_log)
        update_tensor_by_update_lists_dim_1(self.fc2.weight.data, self.fc2_weight_partition,
                                            self.hidden_layer_index_log)

class CNNCifar(nn.Module):
    def __init__(self, model_size1, partition_num=1):
        super(CNNCifar, self).__init__()
        self.partition_num = partition_num
        self.model_size1 = model_size1
        self.temp_hidden_layer_index1 = [i for i in range(self.model_size1)]
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, self.model_size1)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.model_size1, 10)

        if self.partition_num != 1:
            self.partition_dim1 = create_list(self.model_size1, self.partition_num)
            self.hidden_layer_index_log1 = [] # store the log of partition on the first hidden nerons
            self.fc1_weight_partition = []
            self.fc2_weight_partition = []
            self.conv_layers = []



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

    def partition_to_list(self):
        print("Repartition parameters!")
        shuffle(self.temp_hidden_layer_index1)
        self.hidden_layer_index_log1.clear()
        for i in range(self.partition_num):
            current_indexes = torch.tensor(
                self.temp_hidden_layer_index1[i * self.partition_dim1[0]:i * self.partition_dim1[0] + self.partition_dim1[i]])
            self.hidden_layer_index_log1.append(current_indexes)

        # record the weight for each cell
        self.fc1_weight_partition = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.hidden_layer_index_log1)
        self.fc2_weight_partition = partition_FC_layer_by_input_dim_1(
            self.fc2.weight, self.hidden_layer_index_log1)

        # convolutional layers
        dict = self.state_dict()
        split_para_name = ['conv1.weight','conv1.bias','conv2.weight','conv2.bias', 'conv3.weight','conv3.bias']
        conv_layers = {param: dict[param] for param in split_para_name}
        self.conv_layers = [conv_layers for i in range(self.partition_num)]

    def flush(self):
        # update the model based on the collected parameters.
        # Here we have to get around pytorch variable by use variable.data,
        # since leaf variable disallowing in-place operation
        # Note: only global server will call this method to synchronize all the models from edge servers
        # Apart from fc1, fc2, the remaining parameters are aggegated by averaging, i.e., we can call FedAvg
        conv_paras_temp = FedAvg(self.conv_layers)
        self.conv1.weight.data = conv_paras_temp['conv1.weight']
        self.conv1.bias.data = conv_paras_temp['conv1.bias']
        self.conv2.weight.data = conv_paras_temp['conv2.weight']
        self.conv2.bias.data = conv_paras_temp['conv2.bias']
        self.conv3.weight.data = conv_paras_temp['conv3.weight']
        self.conv3.bias.data = conv_paras_temp['conv3.bias']
        # The fully connected layers are updated in the IST way
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_1(self.fc2.weight.data, self.fc2_weight_partition,
                                            self.hidden_layer_index_log1)
        self.conv_layers.clear()
        self.fc1_weight_partition.clear()
        self.fc2_weight_partition.clear()


class CNNFmnist(nn.Module):
    def __init__(self, model_size1, model_size2, partition_num=1):
        super(CNNFmnist, self).__init__()
        self.partition_num = partition_num
        self.model_size1 = model_size1
        self.model_size2 = model_size2
        self.temp_hidden_layer_index1 = [i for i in range(self.model_size1)]
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, self.model_size1)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.model_size1, self.model_size2)
        self.fc3 = nn.Linear(self.model_size2, 10)
        if self.partition_num != 1:
            self.partition_dim1 = create_list(self.model_size1, self.partition_num)
            self.hidden_layer_index_log1 = [] # store the log of partition on the first hidden nerons
            self.fc1_weight_partition = []
            self.fc2_weight_partition = []
            self.conv_layers = []



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def partition_to_list(self):
        print("Repartition parameters!")
        shuffle(self.temp_hidden_layer_index1)
        self.hidden_layer_index_log1.clear()
        for i in range(self.partition_num):
            current_indexes = torch.tensor(
                self.temp_hidden_layer_index1[i * self.partition_dim1[0]:i * self.partition_dim1[0] + self.partition_dim1[i]])
            self.hidden_layer_index_log1.append(current_indexes)

        # record the weight for each cell
        self.fc1_weight_partition = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.hidden_layer_index_log1)
        self.fc2_weight_partition = partition_FC_layer_by_input_dim_1(
            self.fc2.weight, self.hidden_layer_index_log1)

        # convolutional layers
        dict = self.state_dict()
        split_para_name = ['conv1.weight','conv1.bias','conv2.weight','conv2.bias', 'fc3.weight','fc3.bias']
        conv_layers = {param: dict[param] for param in split_para_name}
        self.conv_layers = [conv_layers for i in range(self.partition_num)]

    def flush(self):
        # update the model based on the collected parameters.
        # Here we have to get around pytorch variable by use variable.data,
        # since leaf variable disallowing in-place operation
        # Note: only global server will call this method to synchronize all the models from edge servers
        # Apart from fc1, fc2, the remaining parameters are aggegated by averaging, i.e., we can call FedAvg
        conv_paras_temp = FedAvg(self.conv_layers)
        self.conv1.weight.data = conv_paras_temp['conv1.weight']
        self.conv1.bias.data = conv_paras_temp['conv1.bias']
        self.conv2.weight.data = conv_paras_temp['conv2.weight']
        self.conv2.bias.data = conv_paras_temp['conv2.bias']
        # The fully connected layers are updated in the IST way
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_1(self.fc2.weight.data, self.fc2_weight_partition,
                                            self.hidden_layer_index_log1)
        self.fc3.weight.data = conv_paras_temp['fc3.weight']
        self.fc3.bias.data = conv_paras_temp['fc3.bias']
        self.conv_layers.clear()
        self.fc1_weight_partition.clear()
        self.fc2_weight_partition.clear()
