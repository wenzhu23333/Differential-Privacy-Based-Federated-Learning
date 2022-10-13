import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from utils.language_utils import word_to_indices, letter_to_vec
from utils.sampling import sample_dirichlet_train_data, synthetic_iid, cifar_iid, cifar_noniid, mnist_iid, mnist_noniid
from torch.utils.data import Dataset


def get_dataset(args):
    # mnist dataset: 10 classes, 60000 training examples, 10000 testing examples.
    # synthetic dataset: 10 classes, 100,000 examples.

    dict_party_user, dict_sample_user = {}, {}

    if args.dataset == 'MNIST':
        data_dir = './data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)
    elif args.dataset == 'Synthetic' and args.iid == True:
        data_dir = './data/synthetic/synthetic_x_0.npz'
        synt_0 = np.load(data_dir)
        X = synt_0['x'].astype(np.float64)
        Y = synt_0['y'].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))
        # sample iid data
        dict_party_user, dict_sample_user = synthetic_iid(train_dataset, args.num_users, args.num_samples)
    elif args.dataset == 'Synthetic' and args.iid == False:

        data_dir = './data/synthetic/synthetic_x_0.npz'
        synt_0 = np.load(data_dir)
        X = synt_0['x'].astype(np.float64)
        Y = synt_0['y'].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))
        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)
    elif args.dataset == 'Cifar':
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        test_dataset = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_party_user = cifar_iid(train_dataset, args.num_users)
        else:
            dict_party_user = cifar_noniid(test_dataset, args.num_users)
    elif args.dataset == 'Fashion-Mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        test_dataset = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                             transform=trans_fashion_mnist)
        if args.iid:
            dict_party_user = mnist_iid(train_dataset, args.num_users)
        else:
            dict_party_user = mnist_noniid(test_dataset, args.num_users)
    elif args.dataset == 'Femnist':
        train_dataset = FEMNIST(train=True)
        test_dataset = FEMNIST(train=False)
        dict_party_user = train_dataset.get_client_dic()
        args.num_users = len(dict_party_user)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'Shakespeare':
        train_dataset = ShakeSpeare(train=True)
        test_dataset = ShakeSpeare(train=False)
        dict_party_user = train_dataset.get_client_dic()
        args.num_users = len(dict_party_user)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        train_dataset = []
        test_dataset = []
        dict_party_user, dict_sample_user = {}, {}
        print('+' * 10 + 'Error: unrecognized dataset' + '+' * 10)
    return train_dataset, test_dataset, dict_party_user, dict_sample_user


def exp_details(args):
    print('\nExperimental details:')
    print(f'Model     : {args.model}')
    print(f'Optimizer : sgd')
    print(f'Learning rate: {args.lr}')
    print(f'Global Rounds: {args.epochs}\n')

    print('Federated parameters:')

    print('{} dataset, '.format(args.dataset))
    print(f'Number of users    : {args.num_users}')
    print(f'Local Batch size   : {args.local_bs}')
    print(f'Local Epochs       : {args.local_ep}\n')
    return


class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, train=True, transform=None, target_transform=None, ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/femnist/train",
                                                                                 "./data/femnist/test")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                if i == 100:
                    break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


class ShakeSpeare(Dataset):
    def __init__(self, train=True):
        super(ShakeSpeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/shakespeare/train",
                                                                                 "./data/shakespeare/test")
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                if i == 100:
                    break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data