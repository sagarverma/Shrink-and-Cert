# Shrink & Cert: Bi-level Optimization for Certified Robustness

## Getting started

Let's start by installing all the dependencies. 

`pip3 install -r requirement.txt`

## Training

`python3 train.py --arch cifar_large_model --exp-mode shrink --configs configs/CIFAR10/sac.yml --k 0.20  --seed 1234 --exp-name CIFAR10_cifar_large_model_Unstructured_BiC_K0.01`

`python3 train.py --arch resnet110 --exp-mode shrink --configs configs/CIFAR10/sac.yml --k 0.20  --seed 1234 --exp-name CIFAR10_resnet110_Unstructured_BiC_K0.01`

`python3 train.py --arch resnet50 --exp-mode shrink --configs configs/ImageNet/sac.yml --k 0.20  --seed 1234 --exp-name ImageNet_resnet50_Unstructured_BiC_K0.01`
