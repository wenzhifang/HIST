# Hierarchical Independent Submodel Training

## Requirements
python==3.6.13
torch==1.10.2
torchvision==0.2.0

## Run

Comparison of Hierarchical Federated Submodel Training and HFedAvg training LeNet-5 and MLP on FMNIST(Fashion-MNIST) is produce by:
> python [main_HIST_FMNIST.py](main_HIST_FMNIST.py)

Comparison of Hierarchical Federated Submodel Training and HFedAvg training CNN on CIFAR-10 is produce by:
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


## Ackonwledgements
Acknowledgements give to [Shaoxiong Ji](https://github.com/shaoxiongji/federated-learning/tree/master) and [Binghang Yuan](https://github.com/BinhangYuan/IST_Release) for their sharing on the implementation on FL and IST.
# HIST
