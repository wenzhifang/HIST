# Hierarchical Independent Submodel Training

## Requirements
python==3.6.13
torch==1.10.2
torchvision==0.2.0



# Model Partition

![Journal_HIST](/Users/fang375/Desktop/HIST_CODE_GitHub/Photo/Journal_HIST.jpg)

The key idea behind *HIST* is a hierarchical version of model partitioning, where we divide the global model into disjoint partitions (or submodels) per round so that each cell is responsible for training only one partition of the model, reducing client-side computational/storage costs and overall communication load.

![neural_partition](/Users/fang375/Desktop/HIST_CODE_GitHub/Photo/neural_partition.jpg)

Model partitioning can be achieved by partitioning the hidden neurons of fully connected layers.
For CNNs, we only partition the fully connected layers while the convolutional layers are shared by different cells. In particular, we let the input and output neurons be independent of partition and partition the hidden neurons every two layers. As a result, the parameter volume of each submodel is equal to $1/N$ of that of the full model on average.







## Run

Comparison of Hierarchical Federated Submodel Training and HFedAvg training LeNet-5 and MLP on FMNIST(Fashion-MNIST) is produced by:
> python [main_HIST_FMNIST.py](main_HIST_FMNIST.py)

Comparison of Hierarchical Federated Submodel Training and HFedAvg training CNN on CIFAR-10 is produced by:
> python [main_HIST_Cifar.py](main_HIST_Cifar.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_HIST_Cifar.py \
    --dataset 'cifar' \
    --model 'cnn' \
    --model_size1 500 \
    --num_users 40 \
    --epochs 200 \
    --local_bs 50 \
    --local_ep 2 \ #H=sample_size/local_bs*local_ep
    --num_edge_steps #5 \ E=num_edge_steps
    --num_cells 4
> python main_HIST_FMNIST.py \
    --dataset 'fmnist' \
    --model 'cnn' \
    --model_size1 120 \
    --model_size2 84 \
    --num_users 40 \
    --epochs 200 \
    --local_bs 50 \
    --local_ep 2 \
    --num_edge_steps 5 \
    --iid \ #mixed distribution
    --num_cells 4


## Acknowledgments
Acknowledgments were given to [Shaoxiong Ji](https://github.com/shaoxiongji/federated-learning/tree/master) and [Binghang Yuan](https://github.com/BinhangYuan/IST_Release) for their sharing on the implementation of FL and IST.
