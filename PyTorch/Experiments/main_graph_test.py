# CIFAR-10 using CNN using Pytorch

from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print('GPU used:', torch.cuda.current_device())



# Hyper Params
epochs = 250
batch_size = 100
learning_rate = 0.0005

# CIFAR Dataset
train_dataset = dsets.CIFAR10(root='./data/', 
    train=True, transform=transforms.ToTensor(),
    download=True)

test_dataset = dsets.CIFAR10(root='./data/',
    train=False, transform=transforms.ToTensor())

# Data loader (Input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
    batch_size=batch_size, shuffle=False)

# CNN Model (2 conv layer)
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

    def forward(self, x):
        out = self.layer1(x)
        print(torch.sum(out))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        self.fc = nn.Linear(out.size(1), 10)
        #print(self.fc.weight)
        out = self.fc(out)
        return out

cnn = CNN()
if torch.cuda.is_available():
    cnn.cuda()
#print train_dataset

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

image1 = (torch.LongTensor(1, 3, 32, 32).random_(0, 255)).float()
image2 = (torch.LongTensor(1, 3, 28, 28).random_(0, 255)).float()
images = [image1, image2]
label1 = torch.LongTensor([5])
label2 = torch.LongTensor([7])
labels = [label1, label2]

for epoch in range(2):
    for x, y in zip(images, labels):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()    
        x = Variable(x)
        y = Variable(y)

        optimizer.zero_grad()
        print("\nInput size:", x.size())
        output = cnn(x)
        #print("Output:", output)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print("Loss:", loss.data[0])