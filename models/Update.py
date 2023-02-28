#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple, Gaussian_moment
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, dp_mechanism='no_dp', dp_epsilon=20, dp_delta=1e-5, dp_clip=20):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True)
        self.dp_mechanism = dp_mechanism
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip = dp_clip
        self.idxs = idxs

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)

        loss_client = 0

        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            if self.dp_mechanism != 'no_dp':
                self.clip_gradients(net, len(images))
            optimizer.step()
            scheduler.step()
            # add noises to parameters
            if self.dp_mechanism != 'no_dp':
                self.add_noise(net)
            loss_client = loss.item()

        return net.state_dict(), loss_client, scheduler.get_last_lr()[0]

    def clip_gradients(self, net, batch_size):
        if self.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.perSampleClip(net, self.dp_clip, norm=1)
        elif self.dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            self.perSampleClip(net, self.dp_clip, norm=2)

    def perSampleClip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for factor, grad in zip(per_sample_clip_factor, grad_samples):
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)


    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.args.lr, self.dp_clip, len(self.idxs))
        if self.dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise
        elif self.dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise



class LocalUpdateSerial(object):
    def __init__(self, args, dataset=None, idxs=None, dp_mechanism='no_dp',
                 dp_epsilon=20, dp_delta=1e-5, dp_clip=20):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True)
        self.dp_mechanism = dp_mechanism
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip = dp_clip
        self.idxs = idxs


    def train(self, net, total_sample_number):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)

        for images, labels in self.ldr_train:
            # images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            losses = 0
            index = int(len(images) / self.args.serial_bs)

            # print('total: {}'.format(len(images)))
            for i in range(0, index + 1):
                net.zero_grad()
                start = i * self.args.serial_bs
                end = (i+1) * self.args.serial_bs if (i+1) * self.args.serial_bs < len(images) else len(images)
                # print(end - start)
                if start == end:
                    break
                image_serial_batch, labels_serial_batch \
                    = images[start:end].to(self.args.device), labels[start:end].to(self.args.device)
                log_probs = net(image_serial_batch)
                loss = self.loss_func(log_probs, labels_serial_batch)
                loss.backward()
                if self.dp_mechanism != 'no_dp':
                    self.clip_gradients(net, end - start)
                grads = [param.grad.detach().clone() for param in net.parameters()]
                if i == 0:
                    Total_grads = grads
                    for idx, grad in enumerate(grads):
                        Total_grads[idx] = torch.mul(torch.div((end - start), len(images)), grad)
                else:
                    for idx, grad in enumerate(grads):
                        Total_grads[idx] += torch.mul(torch.div((end - start), len(images)), grad)
                losses += loss.item() * (end - start)
            for i, param in enumerate(net.parameters()):
                param.grad = Total_grads[i]
            optimizer.step()
            scheduler.step()
            # add noises to parameters
            if self.dp_mechanism != 'no_dp':
                self.add_noise(net)

            # print(losses / len(images))

        return net.state_dict(), losses / len(images), scheduler.get_last_lr()[0]

    def clip_gradients(self, net, batch_size):
        if self.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.perSampleClip(net, self.dp_clip, norm=1)
        elif self.dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            self.perSampleClip(net, self.dp_clip, norm=2)

    def perSampleClip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for factor, grad in zip(per_sample_clip_factor, grad_samples):
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.args.lr, self.dp_clip, len(self.idxs))
        if self.dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise
        elif self.dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity,
                                            size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise