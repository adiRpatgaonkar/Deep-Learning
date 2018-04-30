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

# def im2col(image, kernel_size):
#     fh = fw = kernel_size
#     stride = 4
#     im2col_out = torch.FloatTensor()
#     for i in range(0, image.size(2) - kernel_size + 1, stride):
#         for j in range(0, image.size(3) - kernel_size + 1, stride):
#             col_im = image[:, :, i:fh, j:fw]
#             col_im = col_im.contiguous()  # Need to make tensor contiguous to flatten it
#             col_im = col_im.view(col_im.size(0), -1)
#             im2col_out = torch.cat((im2col_out, col_im.t()), 1)  # Cat. as col vector
#             fw += stride  
#         fh += stride
#         fw = kernel_size  # Reset kernel width (Done parsing the width (j) for a certain i)
#     fh = kernel_size  # Reset kernel height (Done parsing the height (i))
#     print(im2col_out.size()) # output im2col tensor
#     return im2col_out

#image1 = (torch.LongTensor(1, 3, 32, 32).random_(0, 255)).float()

# Net
conv1 = nn.Conv2d(3, 2, kernel_size=3, stride=2, pad=1)
# relu1 = nn.ReLU()
# pool1 = nn.MaxPool2d(2)
# conv2 = nn.Conv2d(16, 32, kernel_size=5)
# relu2 = nn.ReLU()
# pool2 = nn.MaxPool2d(2)
# fc = nn.Linear(5*5*32, 10)

# x = torch.unsqueeze(trainset.data[0:2][0])
# HARDCODED VERIFICATION via CS231n
# input = [
#          [[0, 0, 2, 1, 0], [0, 1, 0, 0, 2], [2 ,1, 1, 1, 0], [2, 2, 1, 0, 1], [2, 1, 2, 1, 1]], 
#          [[1, 0, 0, 0, 1], [0, 1, 0, 2, 1], [2, 1, 0, 0, 1], [2, 0, 1, 0, 2], [1, 1, 1, 0, 1]],
#          [[2, 0, 0, 1, 0], [1, 2, 2, 0, 2], [1, 0, 2, 0, 0], [2, 1, 1, 0, 2], [0, 0, 2, 2, 2]]
#         ]
# input = torch.Tensor(input)
# print(input)
x = (torch.LongTensor(1, 3, 5, 5).random_(0, 255)).float()
x1 = (torch.LongTensor([]).float())
#x = cutorch.standardize(x)
out = conv1(input)
print("Out conv1:", out.data.size())
# out = relu1(out)
# print("Out relu1:", out.data.size())
# out = pool1(out)
# print("Out pool1:", out.data.size())
# out = conv2(out)
# print("Out conv2:", out.data.size())
# out = relu2(out)
# print("Out relu2:", out.data.size())
# out = pool2(out)
# print("Out pool2:", out.data.size())
# out = out.data.view(out.data.size(0), -1)
# out = fc(out)
# print("Out fc:", out.data.size())

