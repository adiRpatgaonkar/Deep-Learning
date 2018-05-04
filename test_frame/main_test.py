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
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(5*5*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.data.view(out.data.size(0), -1)
        out = self.fc(out)
        return out
import sys
def im2col(batch, kernel_size, stride, task="conv"):
    # One image @ a time
    batch_i2c = torch.Tensor() 
    # To parse across width and height (temp vars).
    # Keep kernel_size constant
    fh = fw = kernel_size
    for image in batch:
        i2c_per_im = torch.Tensor()
        for i in range(0, image.size(1) - kernel_size + 1, stride):
            for j in range(0, image.size(2) - kernel_size + 1, stride):
                im_col = image[:, i:fh, j:fw]
                im_col = im_col.contiguous()  # tensor must be contiguous to flatten it
                im_col.unsqueeze_(0) # Stretch to 4D tensor
                if task == "conv":
                    # Flatten across 3D space
                    im_col = im_col.view(im_col.size(0), -1)
                elif task == "pooling": 
                    # Flatten across 2D i.e. preserve depth dim
                    im_col = im_col.view(im_col.size(1), -1)
                i2c_per_im = torch.cat((i2c_per_im, im_col.t()), 1)  # Cat as col vector
                fw += stride
            fh += stride
            fw = kernel_size  # Reset kernel width
        fh = kernel_size  # Reset kernel height 
        batch_i2c = torch.cat((batch_i2c, i2c_per_im), 1)
        #print("Bim2c",batch_im2col)
    return batch_i2c

image = (torch.LongTensor(2, 3, 32, 32).random_(0, 255)).float()
im2col_t = im2col(image, 5, 2, task="conv")
print(im2col_t)
#cnn = CNN()
#cnn(image)
#print(cnn.layer1[1].parameters())
#from collections import OrderedDict as OD
#gradients = OD()
#gradients['input'] = torch.randn(2, 16, 28, 28)
#grad = cnn.layer1[1].backward(gradients)
#cnn.layer1[0].backward(grad)

