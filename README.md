# Differential Privacy (DP) Based Federated Learning (FL) 
Everything about DP-based FL you need is here.

（所有你需要的DP-based FL的信息都在这里）
## Code
Tip: the code of this repository is my personal implementation, if there is an inaccurate place please contact me, welcome to discuss with each other. The FL code of this repository is based on this [repository](https://github.com/wenzhu23333/Federated-Learning) .I hope you like it and support it. Welcome to submit PR to improve the  repository.

（提示：本仓库的代码均为本人个人实现，如有不准确的地方请联系本人，欢迎互相讨论。 本仓库的FL代码是基于 [这个仓库](https://github.com/wenzhu23333/Federated-Learning) 实现的，希望大家都能点赞多多支持，欢迎大家提交PR完善，谢谢！ ）

Note that in order to ensure that each client is selected a fixed number of times (to compute privacy budget each time the client is selected), this code uses round-robin client selection, which means that each client is selected sequentially.

(注意，为了保证每个客户端被选中的次数是固定的（为了计算机每一次消耗的隐私预算），本代码使用了Round-robin的选择客户端机制，也就是说每个client是都是被顺序选择的。 )

Important note: The number of FL local update rounds used in this code is all 1, please do not change, once the number of local iteration rounds is changed, the sensitivity in DP needs to be recalculated, the upper bound of sensitivity will be a large value, and the privacy budget consumed in each round will become a lot, so please use the parameter setting of Local epoch = 1.

(重要提示：本代码使用的FL本地更新轮数均为1，请勿更改，一旦更改本地迭代轮数，DP中的敏感度需要重新计算，敏感度上界会是一个很大的值，每一轮消耗的隐私预算会变得很多，所以请使用local epoch = 1的参数设置。)

### Parameter List

**Datasets**: MNIST, Cifar-10, FEMNIST, Fashion-MNIST, Shakespeare.

**Model**: CNN, MLP, LSTM for Shakespeare

**DP Mechanism**: Laplace, Gaussian(Simple Composition), **Todo**: Gaussian(*moments* accountant)

**DP Parameter**: $\epsilon$ and $\delta$

**DP Clip**: In DP-based FL, we usually clip the gradients in training and the clip is an important parameter to calculate the sensitivity.

### Example Mnist Gaussian Mechanism

Experiments: bash run.sh

Drawing: python3 draw.py

![Mnist](mnist_gaussian.png)

### No DP

You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism no_dp

### Laplace Mechanism

This code is based on Simple Composition in DP. In other words, if a client's privacy budget is $\epsilon$ and the client is selected $T$ times, the client's budget for each noising is $\epsilon / T$.

（该代码是基于Simple Composition的，也就是说，如果某个客户端的隐私预算是$\epsilon$，这个客户端被选中$T$次的话，那么该客户端每次加噪使用的预算为$\epsilon / T$ ）

You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism Laplace --dp_epsilon 10 --dp_clip 10

### Gaussian Mechanism

#### Simple Composition

The same as Laplace Mechanism.

You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --dp_clip 10

#### Moments Accountant

See the paper for detailed mechanism. 

Abadi, Martin, et al. "Deep learning with differential privacy." *Proceedings of the 2016 ACM SIGSAC conference on computer and communications security*. 2016.

To do...

## Papers

- Reviews
  - Rodríguez-Barroso, Nuria, et al. "Federated Learning and Differential Privacy: Software tools analysis, the Sherpa. ai FL framework and methodological guidelines for preserving data privacy." *Information Fusion* 64 (2020): 270-292.
- Gaussian Mechanism
  - Wei, Kang, et al. "Federated learning with differential privacy: Algorithms and performance analysis." *IEEE Transactions on Information Forensics and Security* 15 (2020): 3454-3469.
  - Geyer, Robin C., Tassilo Klein, and Moin Nabi. "Differentially private federated learning: A client level perspective." *arXiv preprint arXiv:1712.07557* (2017).
  - Seif, Mohamed, Ravi Tandon, and Ming Li. "Wireless federated learning with local differential privacy." *2020 IEEE International Symposium on Information Theory (ISIT)*. IEEE, 2020.
  - Naseri, Mohammad, Jamie Hayes, and Emiliano De Cristofaro. "Toward robustness and privacy in federated learning: Experimenting with local and central differential privacy." *arXiv e-prints* (2020): arXiv-2009.
  - Truex, Stacey, et al. "A hybrid approach to privacy-preserving federated learning." *Proceedings of the 12th ACM workshop on artificial intelligence and security*. 2019.
  - Triastcyn, Aleksei, and Boi Faltings. "Federated learning with bayesian differential privacy." *2019 IEEE International Conference on Big Data (Big Data)*. IEEE, 2019.
- Laplace Mechanism
  - Wu, Nan, et al. "The value of collaboration in convex machine learning with differential privacy." *2020 IEEE Symposium on Security and Privacy (SP)*. IEEE, 2020.
  - Olowononi, Felix O., Danda B. Rawat, and Chunmei Liu. "Federated learning with differential privacy for resilient vehicular cyber physical systems." *2021 IEEE 18th Annual Consumer Communications & Networking Conference (CCNC)*. IEEE, 2021.
- Other Mechanism
  - Sun, Lichao, Jianwei Qian, and Xun Chen. "Ldp-fl: Practical private aggregation in federated learning with local differential privacy." *arXiv preprint arXiv:2007.15789* (2020).
  - Liu, Ruixuan, et al. "Fedsel: Federated sgd under local differential privacy with top-k dimension selection." *International Conference on Database Systems for Advanced Applications*. Springer, Cham, 2020.
  - Truex, Stacey, et al. "LDP-Fed: Federated learning with local differential privacy." *Proceedings of the Third ACM International Workshop on Edge Systems, Analytics and Networking*. 2020.
  - Zhao, Yang, et al. "Local differential privacy-based federated learning for internet of things." *IEEE Internet of Things Journal* 8.11 (2020): 8836-8853.
  

## Remark
FL-DP Version 2.0(Beta): Using opacus to clip per sample gradient and rewrite code.

Dev分支发布的版本使用了Opacus进行Per Sample Gradient Clip。
