import numpy as np


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(alpha, beta, num_sample):
    dimension = 60
    NUM_CLASS = 10

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    W = np.random.normal(0, alpha, (dimension, NUM_CLASS))
    b = np.random.normal(0, alpha, NUM_CLASS)

    xx = np.random.multivariate_normal(np.array([beta] * dimension), cov_x, num_sample)
    yy = np.zeros(num_sample)

    for j in range(num_sample):
        tmp = np.dot(xx[j], W) + b
        yy[j] = np.argmax(softmax(tmp))

    return xx, yy


def main():
    index = 0
    X, y = generate_synthetic(alpha=1, beta=0, num_sample=100000)
    np.savez('synthetic_x_{}.npz'.format(index), x=X, y=y)


if __name__ == "__main__":
    main()
