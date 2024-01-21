import torch

def create_list(a, b):
    if a % b == 0:
        result = [a // b] * b
    else:
        result = [a // b] * (b - 1) + [a - (a // b) * (b - 1)]
    return result


def partition_FC_layer_by_output_dim_0(tensor, index_buffer):
    #the output dim of tensor will be splitted (Linear Transformation, W_1*X_0)
    results = []
    for current_indexes in index_buffer:
        current_tensor = torch.index_select(tensor, 0, current_indexes)
        results.append(current_tensor)
    return results
    #return a dict containing first-layer weight for each cell

def partition_FC_layer_by_input_dim_1(tensor, index_buffer):
    #the input dim of tensor will be splitted (Linear Transformation W_2*X_1)
    results = []
    for current_indexes in index_buffer:
        current_tensor = torch.index_select(tensor, 1, current_indexes)
        results.append(current_tensor)
    return results
    #return a dict containing final-layer weight for each cell

def partition_FC_layer_by_dim_01(tensor, index_buffer0, index_buffer1):
    '''
    :param tensor:
    :param index_buffer0: dim 0 is the output dim
    :param index_buffer1: dim 1 is the input dim
    :return:
    '''
    assert(len(index_buffer0)==len(index_buffer1))
    split_num = len(index_buffer0)
    results = []
    for i in range(split_num):
        temp_tensor = torch.index_select(tensor, 0, index_buffer0[i])
        current_tensor = torch.index_select(temp_tensor, 1, index_buffer1[i])
        results.append(current_tensor)
    return results


def update_tensor_by_update_lists_dim_0(tensor, update_list, index_buffer):
    assert(len(update_list) == len(index_buffer))
    for i in range(len(update_list)):
        tensor.index_copy_(0, index_buffer[i], update_list[i])


def update_tensor_by_update_lists_dim_1(tensor, update_list, index_buffer):
    assert(len(update_list) == len(index_buffer))
    for i in range(len(update_list)):
        tensor.index_copy_(1, index_buffer[i], update_list[i])

def update_tensor_by_update_lists_dim_01(tensor, update_list, index_buffer0, index_buffer1):
    assert(len(update_list) == len(index_buffer0)
           and len(update_list) == len(index_buffer1))
    for i in range(len(update_list)):
        temp_tensor = torch.index_select(tensor, 0, index_buffer0[i])
        temp_tensor.index_copy_(1, index_buffer1[i], update_list[i])
        tensor.index_copy_(0, index_buffer0[i], temp_tensor)

def update_tensor_by_update_lists_dim_0_overlapping(tensor, update_list, index_buffer):
    assert(len(update_list) == len(index_buffer))
    size = tensor.size()
    temp_tensor = torch.zeros(size)
    count = torch.zeros(size[0])
    for i in range(len(update_list)):
        temp_tensor[index_buffer[i]] = temp_tensor[index_buffer[i]] + update_list[i]
        count[index_buffer[i]] = count[index_buffer[i]] + torch.ones(index_buffer[i].size())
        # Avoid division by zero by setting count to at least 1
    count[count == 0] = 1
    temp_tensor = temp_tensor / count.unsqueeze(1)
    # Copy the values from the temporary tensor back to the original tensor
    tensor.copy_(temp_tensor)

def update_tensor_by_update_lists_dim_1_overlapping(tensor, update_list, index_buffer):
    assert(len(update_list) == len(index_buffer))
    size = tensor.size()
    temp_tensor = torch.zeros(size)
    count = torch.zeros(size[1])
    for i in range(len(update_list)):
        temp_tensor[:,index_buffer[i]] = temp_tensor[:,index_buffer[i]] + update_list[i]
        count[index_buffer[i]] = count[index_buffer[i]] + torch.ones(index_buffer[i].size())
        # Avoid division by zero by setting count to at least 1
    count[count == 0] = 1
    temp_tensor = temp_tensor / count.unsqueeze(0)
    # Copy the values from the temporary tensor back to the original tensor
    tensor.copy_(temp_tensor)

def partition_FC_bias_layer(bias_tensor, index_buffer):
    bias_results = []
    for current_indexes in index_buffer:
        current_bias_tensor = torch.index_select(bias_tensor, 0, current_indexes)
        bias_results.append(current_bias_tensor)
    return bias_results
