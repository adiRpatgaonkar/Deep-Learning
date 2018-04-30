from __future__ import print_function

import torch

import cutorch
import cutorch.nn as nn
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms, see
#from evaluate import *

# if cutorch.gpu_check.available():
#     using_gpu = True
# else:
#     using_gpu = False

# Global vars
global images, ground_truths, outputs, predicted, loss
global train_loader, test_loader

# Hyperparameters
max_epochs = 10
learning_rate = 5e-2
lr_decay = 5e-5

# Get training data for training and Cross validation
# trainset = dsets.CIFAR10(dir="cutorchvision/data",
#                          download=True, train=True,
#                          form="tensor")
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

# train_loader = cutorch.utils.data.DataLoader(data=trainset.data,
#                                             batch_size=100,
#                                             shuffled=True)

# Net
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(5*5*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.data.view(out.data.size(0), -1)
        out = self.fc(out)
        return out


image = (torch.LongTensor(3, 32, 32).random_(0, 255)).float()
cnn = CNN()
cnn(image)
