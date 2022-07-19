import numpy as np

def cal_sensitivity(lr, clip, dataset_size):
#     return 2 * lr * clip / dataset_size
    return 2 * lr * clip

def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)

# todo
def Gaussian_moment(epsilon, delta, sensitivity, size):
    return
