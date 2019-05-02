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


## Implementation Details
Default script will run WRN-28-10: wide residual network with depth=28 and k=10 (widen factor)  
Batch size: 128  

Learning rate settings:

|   epoch   | learning rate |  weight decay | Optimizer | Momentum | Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:--------:|
|   0 ~ 60  |      0.1      |     0.0005    | Momentum  |    0.9   |   true   |
|  61 ~ 120 |      0.02     |     0.0005    | Momentum  |    0.9   |   true   |
| 121 ~ 160 |     0.004     |     0.0005    | Momentum  |    0.9   |   true   |
| 161 ~ 200 |     0.0008    |     0.0005    | Momentum  |    0.9   |   true   |

