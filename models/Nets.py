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
    def __init__(self, dim_in, dim_out):
        super(our_CNN, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.conv1 = nn.Conv1d(self.dim_in, 16, kernel_size=1)  
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(32, 128) 
        self.fc2 = nn.Linear(128, self.dim_out) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Định nghĩa lớp RBM
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def forward(self, v):
        h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return h, torch.bernoulli(h)
    
# Định nghĩa mô hình DBN
class our_DBN(nn.Module):
    def __init__(self, visible_units=21, hidden_units1=64, hidden_units2=64, num_classes=5):
        super(our_DBN, self).__init__()
        
        self.rbm1 = RBM(visible_units=visible_units, hidden_units=hidden_units1)
        self.rbm2 = RBM(visible_units=hidden_units1, hidden_units=hidden_units2)
        rbm_layers = [self.rbm1, self.rbm2]

        self.rbm_layers = nn.ModuleList(rbm_layers)
        self.fc = nn.Linear(rbm_layers[-1].W.shape[1], num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        for rbm in self.rbm_layers:
            x, _ = rbm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class our_RNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers):
        super(our_RNN, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        self.flatten = nn.Flatten()
        self.rnn = nn.RNN(self.dim_in[0], self.dim_hidden, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.dim_hidden, self.dim_out)

    def forward(self, x):
        x = self.flatten(x)
        # batch_size = x.size(0)  # Lấy kích thước batch từ đầu vào
        # h0 = torch.zeros(self.num_layers, batch_size, self.dim_hidden).to(x.device)  # Đảm bảo rằng trạng thái ẩn ban đầu là 2D
        out, _ = self.rnn(x)
        out = self.fc(out)  # chỉ lấy output của lớp cuối cùng
        out = F.softmax(out, dim=1)  # áp dụng softmax
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
