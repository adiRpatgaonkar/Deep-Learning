from __future__ import print_function

import time

import cutorch.nn as nn
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms
from evaluate import *

if cutorch.gpu_check.available():
    using_gpu = True
else:
    using_gpu = False

# Global vars
global curr_time, time2train
global images, ground_truths, outputs, predicted, loss
global train_loader, test_loader

# Hyperparameters
max_epochs = 10
learning_rate = 5e-2
lr_decay = 5e-5
# Get training data for training and Cross validation
trainset = dsets.CIFAR10(dir="cutorchvision/data",
                         download=True, train=True,
                         form="tensor")
# # Data augmentation
# train_dataset = Transforms(dataset=train_dataset,
#                            lr_flip=True, crop=False)
# For testing
# test_dataset = dsets.CIFAR10(directory="cutorchvision/data",
#                              download=True, test=True,
#                              form="tensor")
# # Testing data for validation
# test_loader = cutorch.utils.data.DataLoader(data=test_dataset.data,
#                                             batch_size=10000,
#                                             shuffled=False)

train_loader = cutorch.utils.data.DataLoader(data=trainset.data,
                                            batch_size=100,
                                            shuffled=True)

for epoch in range(5):
    for x, y in train_loader:
        print(type(x), type(y))