# CLS: Cross Labeling Supervision for Semi-Supervised Learning

This is a PyTorch implementation of CLS.

# Usage

### Train

Train the model by 4,000 labeled data of CIFAR-10 dataset:

```python
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 5 --out results/cifar10@4000.5
```



Train the model by 10,000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:

```python
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --dataset cifar100 --num-labeled 10000 --arch wideresnet --batch-size 16 --lr 0.03 --wdecay 0.001 --seed 5 --out results/cifar100@10000.5
```



Monitoring the training progress:

```python
tensorboard --logdir=<your out_dir>
```



# Requirements

- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional) 