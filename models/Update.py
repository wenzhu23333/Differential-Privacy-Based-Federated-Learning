import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule

def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise for CDP-FedAVG-LS Algorithm
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

def laplace_noise(data_shape, sensitivity, global_epoch,epsilon_perRound, device=None):
    """
    Laplace noise
    """
    t = np.random.laplace(0,global_epoch*sensitivity/epsilon_perRound,data_shape)
    return torch.tensor(t).to(device)

def Flatten_gradients(grads):
    # FLatten the grads and restore its original shape
    shape_of_gradients = [x.shape for x in grads]
    return torch.cat([value.flatten() for value in grads]), shape_of_gradients

def Private_gradients(gradient, norm_clip, eps0):
    # "choose at random one coordinate to quantize"
    D_len = len(gradient)
    Z = int(np.random.choice(D_len, 1))
    # clip the gradient
    norm = torch.abs(gradient[Z])
    grads = gradient[Z] / torch.maximum(norm/norm_clip, torch.tensor(1))
    # LDP quantization of a single coordinate
    C = (np.exp(eps0) + 1) / (np.exp(eps0) - 1)
    pr = 0.5 + (grads / (2 * C * norm_clip))
    x = 2 * np.random.binomial(1, pr.detach().cpu().clone(), 1) - 1
    # dequantizater at the aggregator
    H = np.zeros_like(gradient.detach().cpu().clone())
    H[Z] = D_len*x*C*norm_clip
    return H

def reshape_gradients(gradient, shape_of_gradients):
    # Return gradients with original shape
    grads = []
    len_current = 0
    for i in range(0, len(shape_of_gradients)):
        len_of_grad = torch.prod(torch.tensor(shape_of_gradients[i]))
        grads.append(torch.reshape(torch.tensor(gradient[len_current:len_current+len_of_grad]), shape_of_gradients[i]))
        len_current += len_of_grad
    return grads

def perSampleClip(net, batch_size, device, clipping, norm):
    # per sample clip by hand (using opacus)
    grads = [param.grad_sample.detach().clone() for param in net.parameters()]
    for idx in range(batch_size):
        norm_sum = torch.tensor(0.0).to(device)
        for i in range(len(grads)):
            norm_sum += (torch.norm(grads[i][idx].to(torch.float32), p=norm) ** norm)
        norm_sum = torch.sqrt(norm_sum)
        for i in range(len(grads)):
            grads[i][idx] = grads[i][idx] / torch.max(torch.tensor(1),
                                                      norm_sum / (torch.tensor(clipping))).to(device)
    # average per sample gradient after clipping
    for i in range(len(grads)):
        grads[i] = torch.mean(grads[i], dim=0)
    # set back gradient
    for i, param in enumerate(net.parameters()):
        param.grad = grads[i]


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
    def __init__(self, args, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.data_size = len(idxs)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_Laplace(object):
    def __init__(self, args, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        # DPML
        self.current_Tg = 0
        self.enable_lrdecay = True if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device


    def train(self, net):
        net.train()
        # train and update
        self.current_Tg +=1
        if self.enable_lrdecay:
            # self.last_lr = self.lr/(1+self.E*self.current_Tg/50)
            if (self.current_Tg % 10 == 0 and self.current_Tg != 0):
                self.last_lr = self.args.lr / (1 + self.current_Tg / 10)


        optimizer = torch.optim.SGD(net.parameters(), lr=self.last_lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # clip grad + optimize
                if self.enable_dp:
                    perSampleClip(net, len(images), self.args.device, self.clip, norm=1)
                optimizer.step()
                batch_loss.append(loss.item())

                if self.enable_dp:
                    # Add Laplace noise
                    sensitivity = 2 * self.clip * self.last_lr / len(images)
                    state_dict = net.state_dict()  # shallow copy
                    for k, v in state_dict.items():
                        state_dict[k] = state_dict[k] + laplace_noise(v.shape, sensitivity, self.Tg, self.epsilon,
                                                                      device=self.device)
                    net.load_state_dict(state_dict)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_DPSGD(object):
    def __init__(self, args, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.q = args.q
        self.batch_size = 10000
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        # DPSGD
        self.current_Tg = 0
        self.enable_lrdecay = True if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device
        self.delta = 1e-5
        self.sigma = compute_noise(1, self.q, self.epsilon, self.Tg, self.delta, 1e-5)




    def train(self, net):
        net.train()
        # train and update
        self.current_Tg +=1
        if self.enable_lrdecay:
            # self.last_lr = self.lr/(1+self.E*self.current_Tg/50)
            if (self.current_Tg % 10 == 0 and self.current_Tg != 0):
                self.last_lr = self.args.lr / (1 + self.current_Tg / 10)

        optimizer = torch.optim.SGD(net.parameters(), lr=self.last_lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # per sample clip
                perSampleClip(net, len(images), self.args.device, self.args.clip, norm=2)
                for i, param in enumerate(net.parameters()):
                    param.grad += gaussian_noise(param.grad.shape, self.clip, self.sigma / len(images), device=self.device)
                optimizer.step()
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_CODP(object):
    def __init__(self, args, net, dataset=None, idxs=None, ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        # CODP
        self.alpha = args.alpha_CODP
        self.current_Tg = 0
        self.enable_lrdecay = True if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device
        self.last_noise = dict()
        for k, v in net.state_dict().items():
            self.last_noise[k] = torch.zeros_like(v.data, dtype=torch.float32).to(self.device)

    def train(self, net):
        net.train()
        # train and update
        self.current_Tg += 1
        if self.enable_lrdecay:
            # self.last_lr = self.lr/(1+self.E*self.current_Tg/50)
            if (self.current_Tg % 10 == 0 and self.current_Tg != 0):
                self.last_lr = self.args.lr / (1 + self.current_Tg / 10)

        optimizer = torch.optim.SGD(net.parameters(), lr=self.last_lr, momentum=self.args.momentum)

        # RDPFL
        state_dict = net.state_dict()  # shallow copy
        for k, v in state_dict.items():
            state_dict[k] = state_dict[k] - self.alpha * self.last_noise[k]
        net.load_state_dict(state_dict)


        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # clip grad + optimize
                if self.enable_dp:
                    perSampleClip(net, len(images), self.args.device, self.clip, norm=1)
                optimizer.step()
                batch_loss.append(loss.item())

                if self.enable_dp:
                    # Add Laplace noise
                    sensitivity = 2 * self.clip * self.last_lr / len(images)
                    state_dict = net.state_dict()  # shallow copy
                    for k, v in state_dict.items():
                        self.last_noise[k] = laplace_noise(v.shape, sensitivity, self.Tg, self.epsilon,
                                                           device=self.device)
                        state_dict[k] = state_dict[k] + self.last_noise[k]
                    net.load_state_dict(state_dict)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_CODP_gaussian(object):
    def __init__(self, args, net, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.q = args.q
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        # CODP
        self.alpha = args.alpha_CODP
        self.current_Tg = 0
        self.enable_lrdecay = True if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device
        self.delta = 1e-5
        self.sigma = compute_noise(1, self.q, self.epsilon, self.Tg, self.delta, 1e-5)
        self.last_noise = dict()
        for k, v in net.state_dict().items():
            self.last_noise[k] = torch.zeros_like(v.data, dtype=torch.float32).to(self.device)

    def train(self, net):
        net.train()

        # train and update
        self.current_Tg += 1
        if self.enable_lrdecay:
            # self.last_lr = self.lr/(1+self.E*self.current_Tg/50)
            if (self.current_Tg % 5 == 0 and self.current_Tg != 0):
                self.last_lr = self.args.lr / (1 + self.current_Tg / 5)

        optimizer = torch.optim.SGD(net.parameters(), lr=self.last_lr, momentum=self.args.momentum)

        # RDPFL
        state_dict = net.state_dict()  # shallow copy
        for k, v in state_dict.items():
            state_dict[k] = state_dict[k] - self.alpha * self.last_noise[k]
        net.load_state_dict(state_dict)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                # clip grad + optimize
                if self.enable_dp:
                    perSampleClip(net, len(images), self.args.device, self.clip, norm=2)
                optimizer.step()
                batch_loss.append(loss.item())

                if self.enable_dp:
                    # Add Gaussian noise
                    sensitivity = 2 * self.clip * self.last_lr / len(images)

                    state_dict = net.state_dict()  # shallow copy
                    for k, v in state_dict.items():
                        self.last_noise[k] = gaussian_noise(v.shape, sensitivity, self.sigma, device=self.device)
                        state_dict[k] = state_dict[k] + self.last_noise[k]
                    net.load_state_dict(state_dict)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_fDP(object):
    def __init__(self, args, net, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.model = net
        # f-DP
        self.current_Tg = 0
        self.enable_lrdecay = False if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device
        self.delta = 1e-5
        self.q = 1
        self.sigma_fdp = args.sigma_fdp
        self.batch_size = args.local_bs
        self.privacy_engine = PrivacyEngine(
            module=self.model,
            batch_size=self.batch_size,
            sample_size=self.data_size,
            # alpha is not important for f-DP
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=self.sigma_fdp,
            max_grad_norm=self.clip,
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.last_lr, momentum=0.0)
        self.privacy_engine.attach(self.optimizer)


    def train(self, net):

        global_state = net.state_dict()
        self.model.load_state_dict(global_state)
        self.model.train()
        # train and update
        self.current_Tg += 1
        if self.enable_lrdecay:
            if (self.current_Tg % 10 == 0 and self.current_Tg != 0):
                self.last_lr = self.lr / (1 + self.current_Tg / 10)
        for key, value in self.optimizer.param_groups[0].items():
            if key == 'lr':
                self.optimizer.param_groups[0][key] = self.last_lr # opacus?

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_CLDPSGD(object):
    def __init__(self, args, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.batch_size = 10000
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.batch_size, shuffle=True)

        # CLDPSGD
        self.current_Tg = 0
        self.enable_lrdecay = True if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device
        self.eps0 = args.eps0

    def train(self, net):
        net.train()
        # train and update
        self.current_Tg += 1
        if self.enable_lrdecay:
            if (self.current_Tg % 10 == 0 and self.current_Tg != 0):
                self.last_lr = self.args.lr / (1 + self.current_Tg / 10)

        optimizer = torch.optim.SGD(net.parameters(), lr=self.last_lr, momentum=self.args.momentum)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                # log_probs = net(images)
                # loss = self.loss_func(log_probs, labels)
                # loss.backward()
                # perSampleClip(net, len(images), self.args.device, self.args.clip, norm=2)

                # To Do : per Sample flatten
                cldpsgd_loss = 0
                for i in range(0, len(images)):
                    log_probs = net(images[i].unsqueeze(0))
                    loss = self.loss_func(log_probs, labels[i].unsqueeze(0))
                    loss.backward()
                    # per-sample gradient
                    grads = [param.grad.detach().clone() for param in net.parameters()]
                    Flatten_grads, shape_of_grads = Flatten_gradients(grads)
                    Private_grads = Private_gradients(Flatten_grads, self.clip, self.eps0)
                    if i == 0:
                        Total_grads = Private_grads
                    else:
                        Total_grads += Private_grads
                    cldpsgd_loss += float(loss)
                Total_grads /= len(images)
                reshape_grads = reshape_gradients(Total_grads, shape_of_grads)
                # update the model
                for i, param in enumerate(net.parameters()):
                    param.grad.data = reshape_grads[i].to(self.device)
                cldpsgd_loss /= len(images)
                optimizer.step()
                batch_loss.append(cldpsgd_loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_NbaFL(object):
    def __init__(self, args, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        # nbafl
        self.current_Tg = 0
        self.enable_lrdecay = False if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device
        self.delta = args.delta
        self.sigma = self.Tg * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        self.mu_nbafl = args.mu_nbafl


    def train(self, net):
        net.train()
        # train and update
        self.current_Tg += 1
        if self.enable_lrdecay:
            # self.last_lr = self.lr/(1+self.E*self.current_Tg/50)
            if (self.current_Tg % 10 == 0 and self.current_Tg != 0):
                self.last_lr = self.lr / (1 + self.current_Tg / 10)


        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        last_model = copy.deepcopy(net)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                for param, last_param in zip(net.parameters(), last_model.parameters()):
                    param.grad += self.mu_nbafl * (param.data.clone() - last_param.data.clone())

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if self.enable_dp:
            sensitivity = 2.0 * self.clip / self.data_size
            # clip
            norm_2 = torch.tensor(0.0).to(self.device)
            for param in net.parameters():
                norm_2 += (torch.norm(param.data.clone().to(torch.float32), p=2) ** 2)
            norm_2 = torch.sqrt(norm_2).to(self.device)
            for param in net.parameters():
                param.data = param.data / torch.max(torch.tensor(1).to(self.device), norm_2 / torch.tensor(self.clip).to(self.device))
            # Add Gaussian noise
            state_dict = net.state_dict()  # shallow copy
            for k, v in state_dict.items():
                state_dict[k] = state_dict[k] + gaussian_noise(v.shape, sensitivity, self.sigma, device=self.device)
            net.load_state_dict(state_dict)



        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_MGM(object):
    def __init__(self, args, dataset=None, idxs=None,):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        # MGM mechanism
        self.current_Tg = 0
        self.enable_lrdecay = False if args.enable_lrdecay==1 else False
        self.enable_dp = True
        self.lambda_smooth = args.lambda_smooth
        self.lr = args.lr
        self.last_lr = args.lr
        self.clip = args.clip
        self.data_size = len(idxs)
        self.Tg = int(args.epochs * args.frac)
        self.epsilon = args.eps
        self.E = args.local_ep
        self.device = args.device
        self.delta = args.delta
        self.m = 1
        self.n = 0

    def train(self, net):
        net.train()
        # train and update
        self.current_Tg += 1
        if self.enable_lrdecay:
            # self.last_lr = self.lr/(1+self.E*self.current_Tg/50)
            if (self.current_Tg % 10 == 0 and self.current_Tg != 0):
                self.last_lr = self.lr / (1 + self.current_Tg / 10)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.last_lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # Per Sample Clip
                perSampleClip(net, len(images), self.args.device, self.args.clip, norm=2)
                optimizer.step()
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if self.enable_dp:
            sigma = self.generateMGMNoise(net)
            state_dict = net.state_dict()  # shallow copy
            for k, v in state_dict.items():
                state_dict[k] = state_dict[k] + gaussian_noise(v.shape, 1, sigma, device=self.device)
            net.load_state_dict(state_dict)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def generateMGMNoise(self, net):
        # W1,W2,Wu1,Wu2,U2 is Identity matrix
        # m' = m, n' = n
        sensitivity = 2.0 * self.clip * self.last_lr / self.data_size
        m, n = self.m, self.n
        if n == 0:
            for param in net.parameters():
                n += np.prod(list(param.data.shape), axis=0)
            self.n = n
        zeta = -2 * np.log(self.delta) + 2 * np.sqrt(-m * n * np.log(self.delta)) + m * n
        alpha, beta = sensitivity ** 2, sensitivity * zeta * sensitivity
        B = np.square(-beta + np.sqrt(np.square(beta) + 8 * alpha * self.epsilon / self.Tg)) / (4 * np.square(alpha))
        # P1 = P2 = 1, sigma2 = 1
        sigma1 = np.sqrt(n / B)
        return sigma1


