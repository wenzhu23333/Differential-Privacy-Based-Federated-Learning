import matplotlib.pyplot as plt
import matplotlib as mpl

def openfile(filepath):
    file = open(filepath)
    y = []
    while 1:
        line = file.readline()
        if line.rstrip('\n') == '':
            break
        y.append(float(line.rstrip('\n')))
        if not line:
            break
        pass
    file.close()
    return y

if __name__ == '__main__':
    mpl.use('TkAgg')
    plt.figure()
    epsilon_array = ['1.0', '5.0', '10.0', '20.0', '30.0']
    plt.ylabel('test accuracy')
    plt.xlabel('global round')
    for epsilon in epsilon_array:
        y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.format(epsilon))
        plt.plot(range(100), y, label=r'$\epsilon={}$'.format(epsilon))
    y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_no_dp_epsilon_20.dat'.format(epsilon))
    plt.plot(range(100), y, label=r'$\epsilon=+\infty$')
    plt.title('Mnist Gaussian')
    plt.legend()
    plt.savefig('mnist_gaussian.png')

    plt.figure()
    epsilon_array = ['1.0', '5.0', '10.0', '20.0', '30.0']
    plt.ylabel('test accuracy')
    plt.xlabel('global round')
    for epsilon in epsilon_array:
        y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_MA_epsilon_{}.dat'.format(epsilon))
        plt.plot(range(100), y, label=r'$\epsilon={}$'.format(epsilon))
    y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_no_dp_epsilon_20.dat'.format(epsilon))
    plt.plot(range(100), y, label=r'$\epsilon=+\infty$')
    plt.title('Mnist Gaussian Moment Account (q = 0.01)')
    plt.legend()
    plt.savefig('mnist_gaussian_MA.png')

