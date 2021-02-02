# resnet50-imagenet-baseline
Image classification baseline using ResNet50 on ImageNet

## Update

1. I may have found the root cause for the test performance discrepancy.

In this implementation, I happened to use a total batch size equal to 1024 as each process used a batch size of 256 and 4 processes were spawned. 
In the official PyTorch example, each process use `bs=256/N` where `N` is the number of processes (4 here).
It means that I had to either adjust the batch size (i.e. set it to 64 per process) or tune the learning rate accordingly (i.e. set it higher initially, e.g. 0.4 when using 256 images per process)

2. I updated the number of workers to optimize the data throughput. This value is quite specific to the hardware used when training the model.

## Goal

I’m currently interested in reproducing some baseline image classification results using PyTorch.
My goal is to get a `resnet50` model to have a test accuracy as close as the one reported in torchvision [here](https://pytorch.org/vision/0.8/models.html) (**76.15** top 1 accuracy on the official validation set)

In order to do that, I closely follow the setup from the official PyTorch examples [repository](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

However I notice a big gap since I'm only able to obtain **73.12** top 1 accuracy.

This repository is here for reproducibility purposes.
The related PyTorch forums post can be found [here](https://discuss.pytorch.org/t/testing-accuracy-gap-when-training-a-resnet50-on-imagenet-from-scratch/110611)

## Installation

---

The code has been tested on Ubuntu 16.04 with Python 3.7.9

```
$ git clone git@github.com:inzouzouwetrust/resnet50-imagenet-baseline.git
$ cd resnet50-imagenet-baseline && pip install -r requirements.txt
```

Create a data folder and soft link it to where your ImageNet data is located:

```
$ mkdir data
$ ln -s /path/your/imagenet/root $PWD/data/ImageNet
```

## Usage

---

**WARNING**: Make sure to have 4 GPUs with at least 12Gb per GPU on a single node to use DDP.

To train a resnet50 model on ImageNet following [this](https://github.com/pytorch/examples/blob/master/imagenet/main.py) simple setup, use:

```
$ python train.py
```

**NOTE**: The training takes around 1 day when using the proposed setup on 4 Tesla V100 GPUs.

To test the model that you have trained on a specific datetime:

```
$ python inference.py test.checkpoint=data/runs/YYYY-MM-DD_HH-MM-SS
```

