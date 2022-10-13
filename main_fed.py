import copy
import time

import numpy as np
import torch
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from models.Fed import FedAvg, FedWeightAvg
from pathlib import Path
from models.Update import *
from models.Nets import *
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from opacus.grad_sample import GradSampleModule

torch.manual_seed(123)
np.random.seed(123)
torch.cuda.manual_seed_all(123)
torch.cuda.manual_seed(123)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split data for users
    dataset_train, dataset_test, dict_party_user, dict_sample_user = get_dataset(args)

    # build model
    if args.model == 'cnn' and (args.dataset == 'MNIST' or args.dataset == 'Fashion-Mnist'):
        if args.method == 'fDP':
            net_glob = CNNMnist_opacus(data_in=1, data_out=10).to(args.device)
        else:
            net_glob = CNNMnist(data_in=1, data_out=10).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'Cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'Femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'Shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        dataset_train = dataset_train.dataset
        dataset_test = dataset_test.dataset
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    empty_net = net_glob
    print('Model architecture:')

    # use opacus to wrap model to clip per sample gradient
    net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    best_att_acc = 0

    users_classes = []
    for idx in range(0, args.num_users):
        if args.method == 'NoDP':
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx],)
        elif args.method == 'Laplace':
            local = LocalUpdate_Laplace(args=args, dataset=dataset_train, idxs=dict_party_user[idx],)
        elif args.method == 'DPSGD':
            local = LocalUpdate_DPSGD(args=args, dataset=dataset_train, idxs=dict_party_user[idx], )
        elif args.method == 'fDP':
            local = LocalUpdate_fDP(args=args, net=copy.deepcopy(net_glob), dataset=dataset_train,
                                    idxs=dict_party_user[idx],)
        elif args.method == 'NbAFL':
            local = LocalUpdate_NbaFL(args=args, dataset=dataset_train, idxs=dict_party_user[idx],)
        elif args.method == 'CLDPSGD':
            local = LocalUpdate_CLDPSGD(args=args, dataset=dataset_train, idxs=dict_party_user[idx],
                                        )
        elif args.method == 'CODP':
            local = LocalUpdate_CODP(args=args, net=copy.deepcopy(net_glob), dataset=dataset_train,
                                     idxs=dict_party_user[idx], )
        elif args.method == 'CODP_gaussian':
            local = LocalUpdate_CODP_gaussian(args=args, net=copy.deepcopy(net_glob), dataset=dataset_train,
                                              idxs=dict_party_user[idx],)
        elif args.method == 'MGM':
            local = LocalUpdate_MGM(args=args, dataset=dataset_train, idxs=dict_party_user[idx],)
        users_classes.append(local)

    all_clients = list(range(args.num_users))
    model_acc_logs = []
    for iter in range(args.epochs):
        start = time.time()
        loss_locals = []
        dataset_size = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        begin_index = iter % (1 / args.frac)
        idxs_clients = all_clients[int(begin_index * args.num_users * args.frac):
                                   int((begin_index + 1) * args.num_users * args.frac)]

        for idx in idxs_clients:
            local = users_classes[idx]
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            dataset_size.append(len(dict_party_user[idx]))

        # update global weights
        w_glob = FedWeightAvg(w_locals, dataset_size)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
        acc_test, loss_test = test_fun(net_glob, dataset_test, args)
        model_acc_logs.append(acc_test.numpy().tolist())
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        end = time.time()
        print('Round {:3d}, Average training loss {:.3f}, Average test acc {:.3f}, Time {:.3f} s'.format(iter, loss_avg, acc_test, end - start))

    # testing
    net_glob.eval()
    acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
    acc_test, loss_test = test_fun(net_glob, dataset_test, args)
    # experiment setting
    exp_details(args)

    print('Experimental result summary:')
    print("Training accuracy of the joint model: {:.2f}".format(acc_train))
    print("Testing accuracy of the joint model: {:.2f}".format(acc_test))

    results_path = './results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logs = Path(results_path)
    filename = f'{args.method}_{args.dataset}_acc={acc_test:.2f}_eps={args.eps}_clip={args.clip}_client={args.num_users}_alpha={args.alpha}.json'
    data = (model_acc_logs, )
    with (logs / filename).open('w', encoding='utf8') as f:
        json.dump(data, f)

