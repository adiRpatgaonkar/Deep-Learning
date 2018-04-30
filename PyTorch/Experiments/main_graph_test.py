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
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU())       

        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer1_op = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # self.fc = nn.Linear(5*5*32, 10)    

    def forward(self, x):
        out = self.layer0(x)
        print("layer0 ->", torch.sum(out), end=" ")

        if torch.sum(out.data) > 1000:
            out = self.layer1(out)
            print("layer1 ->", end=" ")
        else: 
            out = self.layer1_op(out)
            print("layer1 option ->", end=" ")              

        # print("Output: layer1:", torch.sum(out))
        out = self.layer2(out)
        print("layer2 ->", end= " ")
        out = out.view(out.size(0), -1)
        # Changing linear layer dimenstions on runtime
        self.fc = nn.Linear(out.size(1), 10)
        #print(self.fc.weight)
        out = self.fc(out)
        print("fc")
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
image3 = (torch.LongTensor(1, 3, 50, 50).random_(0, 255)).float()
images = [image1, image2, image3]
label1 = torch.LongTensor([5])
label2 = torch.LongTensor([7])
label3 = torch.LongTensor([3])
labels = [label1, label2, label3]

for epoch in range(1):
    for i, (x, y) in enumerate(zip(images, labels)):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()    
        x = Variable(x)
        y = Variable(y)
        print("Previous:")
        print("Layer 1 weight buffer:", cnn.layer1[0].weight[0][0])
        print("Layer 1 opt weight buffer:", cnn.layer1_op[0].weight[0][0])
        optimizer.zero_grad()
        print("\nImage {}: {}".format(i, x.size()))
        buffers1 = cnn.layer1[0].weight[0][0]
        buffers2 = cnn.layer1_op[0].weight[0][0] 
        output = cnn(x)
        #print("Output:", output)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print("Current:")
        print("Layer 1 weight:", cnn.layer1[0].weight[0][0])
        print("Layer 1 opt weight:", cnn.layer1_op[0].weight[0][0])
        print("Loss:", loss.data[0])
        print("-" * 20)
