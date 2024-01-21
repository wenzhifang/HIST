#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
import matplotlib
matplotlib.use('Agg')
from utils.options import args_parser
from models.Update import Client
from models.Fed import FedAvg
from models.test import test_img
from models.model_dataset_selection import Model_Dataset
from models.model_dispatch_push import dispatch_model_to_edge_server, push_model_to_global_server



if __name__ == '__main__':
    args = args_parser()
    '''
    args.num_users = 60
    args.model = 'mlp'
    args.dataset = 'fmnist'
    args.epochs = 50
    args.local_ep = 1
    args.num_cells = 4
    args.num_edge_steps = 5
    args.model_size = 300
    #args.model_size1 = 120
    #args.model_size2 = 84
    args.local_bs = 50
    '''
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    net_glob_IST, net_glob_fed, dict_users, dataset_train, dataset_test = Model_Dataset(args)

    print(net_glob_IST)
    global_loss_train_list_HIST = []
    global_acc_train_list_HIST = []
    global_acc_test_list_HIST = []
    global_loss_train_list_fed = []
    global_acc_train_list_fed = []
    global_acc_test_list_fed = []

    # initia status training loss & acc, which would be the same for HIST and HFL
    global_acc_train_HIST, global_loss_train_HIST = test_img(net_glob_IST, dataset_train, args)
    global_loss_train_list_HIST.append(global_loss_train_HIST)
    global_acc_train_list_HIST.append(global_acc_train_HIST)
    global_loss_train_list_fed.append(global_loss_train_HIST)
    global_acc_train_list_fed.append(global_acc_train_HIST)
    # initia status testing loss & acc, which would be the same for HIST and HFL
    global_acc_test_HIST, _ = test_img(net_glob_IST, dataset_test, args)
    global_acc_test_list_HIST.append(global_acc_test_HIST)
    global_acc_test_list_fed.append(global_acc_test_HIST)
    print('Initial Global training loss {:.3f}'.format(global_loss_train_HIST))
    print('Initial Global training accuracy {:.3f}'.format(global_acc_train_HIST))
    print('Initial Global test accuracy {:.3f}'.format(global_acc_test_HIST))

    log_name = './save_fmnist/training_log_{}_{}_N{:1d}_H{:1d}_E{:1d}_Epoch{:1d}.txt'.format(args.dataset, args.iid, args.num_cells,
                                                                                args.local_ep, args.num_edge_steps,
                                                                                args.epochs)
    with open(log_name, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting new experiment: {}_{}_{}_N{:1d}_H{:1d}_E{:1d}_Epoch{:1d} \n'.format(args.dataset, args.model, args.iid, args.num_cells,
                                                                               args.local_ep, args.num_edge_steps,
                                                                               args.epochs))

    ##################################################### Training #########################################################
    for iter in range(args.epochs):
        # send the repartitioned models, use these submodel to initialize edge server's models
        net_glob_IST.partition_to_list()
        idxs_cells = [i for i in range(args.num_cells)]
        for idx_c in idxs_cells:
            net_edge_IST = dispatch_model_to_edge_server(args, net_glob_IST, idx_c)
            net_edge_fed = copy.deepcopy(net_glob_fed)
            w_edges_fed = []
            for edge_iter in range(args.num_edge_steps):
                w_locals_IST = []
                w_locals_fed = []
                idxs_users = [i for i in range(args.num_users//args.num_cells)]
                for idx in idxs_users:
                    local = Client(args=args, dataset=dataset_train, idxs=dict_users[idx_c*(args.num_users//args.num_cells)+idx])
                    w, _ = local.train(net=copy.deepcopy(net_edge_IST).to(args.device))
                    w_fed, _ = local.train(net=copy.deepcopy(net_edge_fed).to(args.device))
                    # edge server need to collect models of clients
                    w_locals_IST.append(copy.deepcopy(w))
                    w_locals_fed.append(copy.deepcopy(w_fed))
                # update edge server's model weights
                w_edge_IST = FedAvg(w_locals_IST)
                w_edge_fed = FedAvg(w_locals_fed)
                # edge server update edge model
                net_edge_IST.load_state_dict(w_edge_IST)
                net_edge_fed.load_state_dict(w_edge_fed)
            # Pushing edge models to global server, HIST
            push_model_to_global_server(args, net_glob_IST, net_edge_IST, idx_c)
            w_edges_fed.append(w_edge_fed) # HFL
        # global server update the global model
        net_glob_IST.flush() # HIST, global synchronization
        net_glob_fed.load_state_dict(FedAvg(w_edges_fed)) # HFL, global federated averaging
##################################################### Training #########################################################
        # record global training loss and accuracy for HIST
        global_acc_train_HIST, global_loss_train_HIST = test_img(net_glob_IST, dataset_train, args)
        global_loss_train_list_HIST.append(global_loss_train_HIST)
        global_acc_train_list_HIST.append(global_acc_train_HIST)
        print('Round {:3d}, Global training loss of HIST {:.3f}'.format(iter, global_loss_train_HIST))
        print('Round {:3d}, Global training accuracy of HIST {:.3f}'.format(iter, global_acc_train_HIST))
        # record global testing loss and accuracy for HIST
        global_acc_test_HIST, _ = test_img(net_glob_IST, dataset_test, args)
        global_acc_test_list_HIST.append(global_acc_test_HIST)
        print('Round {:3d}, Global test accuracy of HIST {:.3f}'.format(iter, global_acc_test_HIST))

        # record global training loss and accuracy for Fed
        global_acc_train_fed, global_loss_train_fed = test_img(net_glob_fed, dataset_train, args)
        global_loss_train_list_fed.append(global_loss_train_fed)
        global_acc_train_list_fed.append(global_acc_train_fed)
        print('Round {:3d}, Global training loss of HFL {:.3f}'.format(iter, global_loss_train_fed))
        print('Round {:3d}, Global training accuracy of HFL {:.3f}'.format(iter, global_acc_train_fed))
        # record global testing loss and accuracy for Fed
        global_acc_test_fed, _ = test_img(net_glob_fed, dataset_test, args)
        global_acc_test_list_fed.append(global_acc_test_fed)
        print('Round {:3d}, Global test accuracy of HFL {:.3f}'.format(iter, global_acc_test_fed))

        with open(log_name, 'a') as log_file:
            # record global training loss and accuracy for HIST
            log_file.write('Round {:3d}, Global training loss of HIST {:.3f}\n'.format(iter, global_loss_train_HIST))
            log_file.write('Round {:3d}, Global training accuracy of HIST {:.3f}\n'.format(iter, global_acc_train_HIST))
            log_file.write('Round {:3d}, Global test accuracy of HIST {:.3f}\n'.format(iter, global_acc_test_HIST))
            log_file.write('Round {:3d}, Global training loss of HFL {:.3f}\n'.format(iter, global_loss_train_fed))
            log_file.write('Round {:3d}, Global training accuracy of HFL {:.3f}\n'.format(iter, global_acc_train_fed))
            log_file.write('Round {:3d}, Global test accuracy of HFL {:.3f}\n'.format(iter, global_acc_test_fed))

    data_to_save = {
        'args': args,
        'global_loss_train_list_HIST': global_loss_train_list_HIST,
        'global_acc_train_list_HIST': global_acc_train_list_HIST,
        'global_acc_test_list_HIST': global_acc_test_list_HIST,
        'global_loss_train_list_fed': global_loss_train_list_fed,
        'global_acc_train_list_fed': global_acc_train_list_fed,
        'global_acc_test_list_fed': global_acc_test_list_fed
    }
    # Save the dictionary containing the data to a single file
    torch.save(data_to_save, 'save_fmnist/Results_{}_{}_N{:1d}_H{:1d}_E{:1d}_Epoch{:1d}.pt'.format(args.dataset, args.iid, args.num_cells, args.local_ep, args.num_edge_steps, args.epochs))

''' 
python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'mlp' \
    --model_size 300 \
    --num_users 60 \
    --epochs 50 \
    --num_cells 3 \
    --local_bs 50 \
    --local_ep 1 \
    --num_edge_steps 5

python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --num_channels 1 \
    --model 'mlp' \
    --model_size 300 \
    --num_users 60 \
    --epochs 50 \
    --num_cells 3 \
    --local_bs 50 \
    --local_ep 2 \
    --num_edge_steps 5

python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'mlp' \
    --model_size 300 \
    --num_users 60 \
    --epochs 50 \
    --num_cells 3 \
    --local_bs 100 \
    --local_ep 1 \
    --num_edge_steps 5 \
    --iid
'''
# for CNN
'''
python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'cnn' \
    --model_size1 120 \
    --model_size2 84 \
    --num_users 40 \
    --epochs 200 \
    --local_bs 50 \
    --local_ep 2 \
    --num_edge_steps 5 \
    --iid \
    --num_cells 2
    
python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'cnn' \
    --model_size1 120 \
    --model_size2 84 \
    --num_users 40 \
    --epochs 200 \
    --local_bs 50 \
    --local_ep 2 \
    --num_edge_steps 5 \
    --num_cells 2
    
python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'cnn' \
    --model_size1 120 \
    --model_size2 84 \
    --num_users 40 \
    --epochs 200 \
    --local_bs 50 \
    --local_ep 2 \
    --num_edge_steps 5 \
    --iid \
    --num_cells 4
    
python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'cnn' \
    --model_size1 120 \
    --model_size2 84 \
    --num_users 40 \
    --epochs 200 \
    --local_bs 50 \
    --local_ep 2 \
    --num_edge_steps 5 \
    --num_cells 4
'''
""" 
python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'mlp' \
    --model_size 300 \
    --num_users 60 \
    --num_cells 4 \
    --local_bs 25 \
    --local_ep 1 \
    --num_edge_steps 5 \
    --epochs 50 \
    
python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'mlp' \
    --model_size 300 \
    --num_users 60 \
    --num_cells 4 \
    --local_bs 25 \
    --iid
    --local_ep 1 \
    --num_edge_steps 5 \
    --epochs 50 \
"""
