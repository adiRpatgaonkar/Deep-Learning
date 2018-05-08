from __future__ import print_function

import cutorch
import cutorch.nn as nn
import cutorch.nn.functionals as F
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms

# if cutorch.gpu_check.available():
#     using_gpu = True
# else:
#     using_gpu = False

# Hyperparameters
max_epochs = 10
learning_rate, lr_decay = 5e-2, 5e-5
reg = 1e-3
batch_size = 100

# Get training data for training and Cross validation
trainset = dsets.CIFAR10(dir="cutorchvision/data",
                         download=True, train=True,
                         form="tensor")
# Data augmentation
trainset = Transforms(dataset=trainset,
                      lr_flip=True, crop=False)
# For testing
testset = dsets.CIFAR10(dir="cutorchvision/data",
                        download=True, test=True,
                        form="tensor")
# Testing data for validation
test_loader = cutorch.utils.data.DataLoader(data=testset.data,
                                            batch_size=10000,
                                            shuffled=False)

train_loader = cutorch.utils.data.DataLoader(data=trainset.data,
                                             batch_size=batch_size,
                                             shuffled=True)


class CNN(nn.Module):
    # Net
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
        self.fc = nn.Sequential(
            nn.Linear(5*5*32, 10),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.data.view(out.data.size(0), -1)
        out = self.fc(out)
        return out

import torch
image = (torch.LongTensor(100, 3, 32, 32).random_(0, 255)).float()
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = cutorch.optim.Optimizer(cnn, lr=learning_rate, lr_decay=lr_decay, reg=reg)
'''
cnn.train()
outputs = cnn(image)
print("Net-out:", outputs.data, outputs.data.size())
print(cnn.layer1[1].parameters(), cnn.forward_path)
# Backprop testing
# Dummy grads
from collections import OrderedDict as OD
gradients = OD()
gradients['in'] = torch.randn(100, 16, 14, 14)
grad = cnn.layer1[3].backward(gradients)
grad = cnn.layer1[2].backward(grad)
grad = cnn.layer1[1].backward(grad)
grad = cnn.layer1[0].backward(grad)
cnn.eval()
print(outputs.data)
'''
for epoch in range(2):
   for i, (images, labels) in enumerate(train_loader):
        cnn.train()
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print ('Epoch [%d/%d], Iter[%d/%d] Loss: %.4f'
                %(epoch + 1, 2, i + 1, len(trainset.data)//batch_size, loss.data)) 
