import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="Global rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users in FL: K")
    parser.add_argument('--num_samples', type=int, default=100,
                        help="number of training samples selected from each local training set")
    parser.add_argument('--alpha', type=float, default=0.3, help="level of non-iid data distribution: alpha")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='Fashion-Mnist', help="name of dataset")
    parser.add_argument('--iid', default=False, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes in the dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    # DP Setting
    parser.add_argument('--method', type=str, default='NoDP', choices=[
        'Laplace', 'DPSGD', 'NbAFL', 'fDP', 'CLDPSGD', 'CODP', 'CODP_gaussian', 'NoDP', 'MGM'
    ])
    parser.add_argument('--eps', type=float, default=10)
    parser.add_argument('--clip', type=float, default=20)
    parser.add_argument('--lambda_smooth', type=float, default=0.5)
    parser.add_argument('--enable_lrdecay', type=int, default=1, help='1 means true, 0 means false')
    parser.add_argument('--delta', type=float, default=1e-5, help='1e-2 for nbafl, 1e-5 for fDP, DPSGD')
    parser.add_argument('--q', type=float, default=0.05, help='sample rate for DPSGD and CODP gaussian')
    parser.add_argument('--eps0', type=float, default=7.94, help='eps for CLDPSGD')
    parser.add_argument('--mu_nbafl', type=float, default=0.001, help='hyper param for nbafl')
    parser.add_argument('--alpha_CODP', type=float, default=1, help='hyper param for CODP')
    parser.add_argument('--sigma_fdp', type=float, default=1, help='hyper param for fDP')
    args = parser.parse_args()
    return args
