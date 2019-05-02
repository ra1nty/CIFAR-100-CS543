# CIFAR-100-CS543
How to ace on a custom CIFAR-100 dataset (CS543 UIUC)

## WRNs:
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146 "Original Paper")  

My implementation in ```wrn.py```

## AutoAugment:
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501 "Original Paper")

In ```autoaugment.py```, there are 25 subpolicies trained from a reduced CIFAR-10 dataset, we use those subpolicies in our CIFAR-100 training.

Implementation ported from https://github.com/DeepVoltaire/AutoAugment

## Cutout:
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552 "Original Paper")  

My implementation in ```cutout.py```
