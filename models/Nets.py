#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from opacus.layers import DPLSTM

class our_MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden_list, dim_out):
        super(our_MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden_list = dim_hidden_list
        self.dim_out = dim_out

        # Khởi tạo lớp input
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_hidden_list[0])])

        # Thêm các lớp ẩn
        for i in range(len(dim_hidden_list) - 1):
            self.layers.append(nn.Linear(dim_hidden_list[i], dim_hidden_list[i+1]))

        # Lớp output
        self.layers.append(nn.Linear(dim_hidden_list[-1], dim_out))

        # Activation function
        self.relu = nn.ReLU()

        # Softmax (nếu bạn muốn softmax đầu ra)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        # Truyền tiến qua các lớp ẩn
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)

        # Lớp output (không cần áp dụng softmax nếu sử dụng cross-entropy loss)
        x = self.layers[-1](x)
        return x

class our_CNN(nn.Module):
    def __init__(self, dim_in, dim_hidden_list, dim_out):
        super(our_CNN, self).__init__()

        # Assuming dim_in is a tuple (channels, height, width)
        self.channels, self.height, self.width = dim_in

        layers = []
        in_channels = self.channels

        for out_channels, kernel_size, stride in dim_hidden_list:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            self.output_size = self._get_conv_output((1, *dim_in))

        self.fc = nn.Linear(self.output_size, dim_out)

    def _get_conv_output(self, shape):
        input = torch.rand(*shape)
        output = self.features(input)
        return int(torch.numel(output) / output.shape[0])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        _, h = self.v_to_h(v)
        _, v = self.h_to_v(h)
        return v
    
class DBN(nn.Module):
    def __init__(self, dim_in, dim_hidden_list, dim_out):
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList()
        self.output_layer = nn.Linear(dim_hidden_list[-1], dim_out)

        # Initialize the RBMs
        previous_layer_size = dim_in
        for layer_size in dim_hidden_list:
            self.rbm_layers.append(RBM(previous_layer_size, layer_size))
            previous_layer_size = layer_size

    def forward(self, x):
        for rbm in self.rbm_layers:
            _, x = rbm.v_to_h(x)
        x = self.output_layer(x)
        return x

class our_RNN(nn.Module):
    def __init__(self, dim_in, dim_hidden_list, dim_out):
        super(our_RNN, self).__init__()
        
        # We will use the first element of dim_hidden_list as the RNN hidden size
        # RNN layers require input of shape (seq_len, batch, input_size)
        self.rnn = nn.RNN(input_size=dim_in, hidden_size=dim_hidden_list[0], num_layers=len(dim_hidden_list), batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(dim_hidden_list[-1], dim_out)

    def forward(self, x):
        # x should be of shape (batch, seq_len, input_size)
        # Get the outputs and the last hidden state
        rnn_out, _ = self.rnn(x)
        
        # We take the output from the last time step for final prediction
        # rnn_out shape is (batch, seq_len, hidden_size)
        last_time_step_out = rnn_out[:, -1, :]
        
        # Output layer
        out = self.fc(last_time_step_out)
        return out

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = x.view(x.shape[0], -1)

        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, args.num_classes)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNFemnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)


class CharLSTM(nn.Module):
    def __init__(self):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = DPLSTM(8, 256, 2, batch_first=True)
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])
