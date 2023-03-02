import numpy as np
from tensorflow_privacy.compute_noise_from_budget_lib import compute_noise


def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size


#     return 2 * lr * clip

def cal_sensitivity_MA(lr, clip, dataset_size):
    return lr * clip / dataset_size


# def Laplace(epsilon, sensitivity, size):
#     noise_scale = sensitivity / epsilon
#     return np.random.laplace(0, scale=noise_scale, size=size)

def Laplace(epsilon):
    return 1 / epsilon


def Gaussian_Simple(epsilon, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def Gaussian_MA(epsilon, delta, q, epoch):
    return compute_noise(1, q, epsilon, epoch, delta, 1e-5)
