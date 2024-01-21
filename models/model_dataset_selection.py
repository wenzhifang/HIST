from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid, mnist_mix, cifar_iid, cifar_noniid, cifar_mix
from models.Nets import CNNCifar, CNNFmnist, MLP

def Model_Dataset(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            #dict_users = mnist_mix(dataset_train, args.num_users, args.num_cells)
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../data/fmnist', train=False, download=True, transform=trans_fmnist)
        #fmnist and mnist share the same partition strategy
        if args.iid:
            #dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_mix(dataset_train, args.num_users, args.num_cells)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            #dict_users = cifar_iid(dataset_train, args.num_users)
            dict_users = cifar_mix(dataset_train, args.num_users, args.num_cells)
        else:
            #exit('Error: only consider IID setting in CIFAR10')
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob_IST = CNNCifar(model_size1=args.model_size1, partition_num=args.num_cells).to(args.device)
        net_glob_fed = CNNCifar(model_size1=args.model_size1).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob_IST = CNNFmnist(model_size1=args.model_size1, model_size2=args.model_size2, partition_num=args.num_cells).to(args.device)
        net_glob_fed = CNNFmnist(model_size1=args.model_size1, model_size2=args.model_size2).to(args.device)
    elif args.model == 'mlp':
        net_glob_IST = MLP(dim_in=784, model_size=args.model_size, dim_out=10, partition_num=args.num_cells).to(args.device)
        net_glob_fed = MLP(dim_in=784, model_size=args.model_size, dim_out=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    return net_glob_IST, net_glob_fed, dict_users, dataset_train, dataset_test


