import copy
from models.Nets import CNNCifar, CNNFmnist, MLP

# global server dispatch model to edge servers
def dispatch_model_MLP_to_edge_server(edge_model, global_model, idx_c):
    edge_model.fc1.weight.data = global_model.fc1_weight_partition[idx_c].clone().detach()
    edge_model.fc2.weight.data = global_model.fc2_weight_partition[idx_c].clone().detach()

def dispatch_model_CNN_Cifar_to_edge_server(edge_model, global_model, idx_c):
    edge_model.fc1.weight.data = global_model.fc1_weight_partition[idx_c].clone().detach()
    edge_model.fc2.weight.data = global_model.fc2_weight_partition[idx_c].clone().detach()

    edge_model.conv1.load_state_dict(copy.deepcopy(global_model.conv1.state_dict()))
    edge_model.conv2.load_state_dict(copy.deepcopy(global_model.conv2.state_dict()))
    edge_model.conv3.load_state_dict(copy.deepcopy(global_model.conv3.state_dict()))

def dispatch_model_CNN_fmnist_to_edge_server(edge_model, global_model, idx_c):
    edge_model.fc1.weight.data = global_model.fc1_weight_partition[idx_c].clone().detach()
    edge_model.fc2.weight.data = global_model.fc2_weight_partition[idx_c].clone().detach()

    edge_model.fc3.load_state_dict(copy.deepcopy(global_model.fc3.state_dict()))
    edge_model.conv1.load_state_dict(copy.deepcopy(global_model.conv1.state_dict()))
    edge_model.conv2.load_state_dict(copy.deepcopy(global_model.conv2.state_dict()))

def dispatch_model_to_edge_server(args, global_model, idx_c):
    if args.model == 'mlp':
        edge_model = MLP(dim_in=784, model_size=global_model.partition_dim[idx_c], dim_out=10).to(
            args.device)
        dispatch_model_MLP_to_edge_server(edge_model, global_model, idx_c)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        edge_model = CNNCifar(model_size1=global_model.partition_dim1[idx_c],\
                                partition_num=args.num_cells).to(args.device)
        dispatch_model_CNN_Cifar_to_edge_server(edge_model, global_model, idx_c)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        edge_model = CNNFmnist(model_size1=global_model.partition_dim1[idx_c], model_size2=args.model_size2, \
                                partition_num=args.num_cells).to(args.device)
        dispatch_model_CNN_fmnist_to_edge_server(edge_model, global_model, idx_c)
    else:
        exit('Error in dataset and model')
    return edge_model

# edge servers push models to the global server

def push_model_MLP_to_global_server(global_model, edge_model, idx_c):
    global_model.fc1_weight_partition[idx_c] = copy.deepcopy(edge_model.fc1.weight.data)
    global_model.fc2_weight_partition[idx_c] = copy.deepcopy(edge_model.fc2.weight.data)

def push_model_CNN_Cifar_to_global_server(global_model, edge_model, idx_c):
    global_model.fc1_weight_partition[idx_c] = copy.deepcopy(edge_model.fc1.weight.data)
    global_model.fc2_weight_partition[idx_c] = copy.deepcopy(edge_model.fc2.weight.data)
    # Convolutional layers
    dict = edge_model.state_dict()
    split_para_name = ['conv1.weight','conv1.bias','conv2.weight','conv2.bias', 'conv3.weight','conv3.bias']
    conv_layer_params = {param: dict[param] for param in split_para_name}
    global_model.conv_layers[idx_c] = copy.deepcopy(conv_layer_params)

def push_model_CNN_fmnist_to_global_server(global_model, edge_model, idx_c):
    global_model.fc1_weight_partition[idx_c] = copy.deepcopy(edge_model.fc1.weight.data)
    global_model.fc2_weight_partition[idx_c] = copy.deepcopy(edge_model.fc2.weight.data)
    # Convolutional layers
    dict = edge_model.state_dict()
    split_para_name = ['conv1.weight','conv1.bias','conv2.weight','conv2.bias', 'fc3.weight','fc3.bias']
    conv_layer_params = {param: dict[param] for param in split_para_name}
    global_model.conv_layers[idx_c] = copy.deepcopy(conv_layer_params)

def push_model_to_global_server(args, global_model, edge_model, idx_c):
    if args.model == 'mlp':
        push_model_MLP_to_global_server(global_model, edge_model, idx_c)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        push_model_CNN_Cifar_to_global_server(global_model, edge_model, idx_c)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        push_model_CNN_fmnist_to_global_server(global_model, edge_model, idx_c)
    else:
        exit("error: unrecognized model and dataset")
