import numpy as np
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch 

class BNaT(Dataset):
    def __init__(self, csv_file, train=True, transform=None, target_transform=None):
        super(BNaT, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        # Đọc dữ liệu từ file CSV
        data = read_dataset(csv_file)
        if self.train:
            # Xử lý dữ liệu training
            self.data = data.iloc[:, :-1].values.astype(np.float32)  # Lấy tất cả các cột trừ cột label
            self.label = data.iloc[:, -1].values  # Lấy cột label
        else:
            # Xử lý dữ liệu test
            self.data = data.iloc[:, :-1].values.astype(np.float32)  # Lấy tất cả các cột trừ cột label
            self.label = data.iloc[:, -1].values  # Lấy cột label

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = img.reshape(21, 1)  # Định dạng lại kích thước ảnh theo yêu cầu của bạn
        
        # Áp dụng các biến đổi nếu có
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.data)

# Hàm transform không cần thiết nếu bạn không muốn xử lý trước dữ liệu

# Ví dụ về một hàm transform đơn giản
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample)

def read_BNaT(train_dir, test_dir):
    transform = ToTensor()
    train = BNaT(csv_file=train_dir, train=True, transform=transform)

    test = BNaT(csv_file=test_dir, train=False, transform=transform)

    return train, test

def read_dataset(raw_file):
    pd_dataset = read_data(raw_file)
    nomial(pd_dataset)
    num_features = ["duration", "protocol_type", "service",	"src_bytes", "dst_bytes", "flag", "count", "srv_count", "serror_rate", "same_srv_rate", "diff_srv_rate", "srv_serror_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_serror_rate", "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate"]      
    pd_dataset[num_features] = pd_dataset[num_features].astype(float)
    pd_dataset[num_features] = MinMaxScaler().fit_transform(pd_dataset[num_features].values)
    labels_test  = pd_dataset["label"].copy()
    labels_test[labels_test == 'Normal'] = 0
    labels_test[labels_test == 'BP'] = 1
    labels_test[labels_test == 'DoS'] = 2
    labels_test[labels_test == 'FoT'] = 3
    labels_test[labels_test == 'MitM'] = 4
    pd_dataset['label'] = labels_test 
    # labels_test = np.array(labels_test, dtype=int)
    return pd_dataset


def read_data(filename):
    col_names = ["duration", "protocol_type", "service",	"src_bytes", "dst_bytes", "flag", "count", "srv_count", "serror_rate", "same_srv_rate", "diff_srv_rate", "srv_serror_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_serror_rate", "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate", "label"]
    dataset = pd.read_csv(filename, header = None, names = col_names)
    return dataset      

def nomial(dataset):
    protocol1 = dataset['protocol_type'].copy()
    protocol_type = ["tcp", "udp", "icmp"]
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
    dataset['protocol_type'] = protocol1

    service1 = dataset['service'].copy()
    service_type = ["other", "private", "ecr_i", "urp_i", "urh_i", "red_i", "eco_i", "tim_i", "oth_i", "domain_u", "tftp_u", "ntp_u", "IRC", 
                "X11", "Z39_50", "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "echo", "efs", "exec", 
                "finger", "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784", "http_443", "http_8001", "icmp", "imap4",
                "iso_tsap", "klogin", "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat",
                "nnsp", "nntp", "pm_dump", "pop_2", "pop_3", "printer", "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", 
                "supdup", "systat", "telnet", "time", "uucp", "uucp_path", "vmnet", "whois"]
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
    dataset['service'] = service1

    flag1 = dataset['flag'].copy()
    flag_type = ["SF", "S0", "S1", "S2", "S3", "REJ", "RSTOS0", "RSTO", "RSTR", "SH", "RSTRH", "SHR", "OTH"]
    for i in range(len(flag_type)):
        flag1[flag1 == flag_type[i]] = i
    dataset['flag'] = flag1