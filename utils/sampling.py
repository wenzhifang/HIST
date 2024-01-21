#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, 60000 // num_users // 2
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_mix(dataset, num_users, num_cells):
    """
    IID intra-cell, non-IID inter-cell
    :param dataset:
    :param num_users:
    :param num_cells:
    :return:
    """
    num_items = len(dataset)//num_cells
    all_idxs = [i for i in range(len(dataset))]
    labels = dataset.train_labels.numpy()
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_cells):
        # partition $num_cells$ cell datasets 
        cell_data_idx = set(np.random.choice(all_idxs, num_items, replace=False))
        # the remaining part is left for other cells
        all_idxs = list(set(all_idxs) - cell_data_idx)
        num_shards, num_imgs = num_users*2//num_cells, num_items // (num_users*2 // num_cells)
        # data shards for the current cell
        idx_shard = [k for k in range(num_shards)]
        idxs = list(cell_data_idx) #np.arange(i*num_items:(i+1)*num_items)
        labels_cell = labels[idxs] # labels[list(dict_cells)]
        print('mix: i: {:1d}'.format(i))
        # sort labels
        idxs_labels = np.vstack((idxs, labels_cell))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for j in range(num_users//num_cells):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i*(num_users//num_cells)+j] = np.concatenate((dict_users[i*(num_users//num_cells)+j],
                                                                         idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)    
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

'''
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset: CIFAR dataset
    :param num_users: Number of users (clients)
    :return: Dictionary with keys as user IDs and values as data indices
    """
    # CIFAR has 10 classes, adjust these numbers as needed
    num_shards, num_imgs_per_shard = 200, 250  # Adjust based on your CIFAR dataset variant and requirement
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs_per_shard)
    labels = np.array(dataset.targets)  # Assuming dataset.targets exists and is a list of labels

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # Each user gets 2 shards
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs_per_shard:(rand + 1) * num_imgs_per_shard]), axis=0)

    return dict_users
'''
'''
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 20, 2500
    num_shards, num_imgs = num_users * 2, 50000 // num_users // 2
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, int(num_shards / num_users), replace=False))
        # rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
'''
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset: CIFAR dataset
    :param num_users: Number of users (clients)
    :return: Dictionary with keys as user IDs and values as data indices
    """
    # CIFAR has 10 classes, adjust these numbers as needed
    num_shards, num_imgs_per_shard = num_users * 2, 50000 // num_users // 2  # Adjust based on your CIFAR dataset variant and requirement
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(50000)
    labels = np.array(dataset.targets)  # Assuming dataset.targets exists and is a list of labels

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # Each user gets 2 shards
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs_per_shard:(rand + 1) * num_imgs_per_shard]), axis=0)

    return dict_users


def cifar_mix(dataset, num_users, num_cells):
    """
    Sample III intra-cell, non-IID inter-cell client data from CIFAR10 dataset
    :param dataset: CIFAR10 dataset
    :param num_users: Total number of users (clients)
    :param num_cells: Number of cells (groups of users)
    :return: Dictionary with keys as user IDs and values as data indices
    """
    num_items = len(dataset.data) // num_cells
    all_idxs = [i for i in range(len(dataset.data))]
    labels = np.array(dataset.targets)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for i in range(num_cells):
        # Partition 'num_cells' cell datasets
        cell_data_idx = set(np.random.choice(all_idxs, num_items, replace=False))
        # The remaining part is left for other cells
        all_idxs = list(set(all_idxs) - cell_data_idx)
        num_shards, num_imgs = num_users*2 // num_cells, num_items // (num_users*2 // num_cells)
        # Data shards for the current cell
        idx_shard = [k for k in range(num_shards)]
        idxs = list(cell_data_idx)
        labels_cell = labels[idxs]
        print('Cell: {:1d}'.format(i))
        # Sort labels
        idxs_labels = np.vstack((idxs, labels_cell))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # Divide and assign
        for j in range(num_users // num_cells):
            user_id = i * (num_users // num_cells) + j
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[user_id] = np.concatenate(
                    (dict_users[user_id],
                     idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True)
    num_users, num_cells = 40, 4
    dict_users = cifar_mix(dataset_train, num_users, num_cells)
