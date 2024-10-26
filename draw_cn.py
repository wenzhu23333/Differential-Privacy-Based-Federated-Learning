import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

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
    myfont = fm.FontProperties(fname='/Users/yangwenzhuo/Library/Fonts/SimHei.ttf')
    with PdfPages('dpfl_example.pdf') as pdf:
        plt.figure()
        epsilon_array = ['1.0', '5.0', '10.0', '20.0', '30.0']
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.ylabel('模型准确率', fontproperties=myfont, fontsize=20)
        plt.xlabel('训练轮次', fontproperties=myfont, fontsize=20)
        for epsilon in epsilon_array:
            y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.format(epsilon))
            plt.plot(range(100), y, label=r'$\epsilon={}$'.format(epsilon))
        y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_no_dp_epsilon_20.dat'.format(epsilon))
        plt.plot(range(100), y, label=r'$\epsilon=+\infty$')
        # plt.title('MNIST数据集', fontproperties=myfont)
        # plt.legend()
        plt.legend(handlelength=1, ncol=2, loc='best', fontsize=15, columnspacing=0.5)
        plt.grid()
        pdf.savefig()
        # plt.savefig('xxx.png')


