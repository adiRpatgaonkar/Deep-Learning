# CIFAR-10 using CNN using Pytorch

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

if torch.cuda.is_available():
    torch.cuda.set_device(1)
    print 'GPU used:', torch.cuda.current_device()

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
        self.fc = nn.Linear(5*5*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cnn = CNN()
if torch.cuda.is_available():
    cnn.cuda()
#print train_dataset

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter[%d/%d] Loss: %.4f'
            %(epoch + 1, epochs, i + 1, len(train_dataset)//batch_size, loss.data[0]))

cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    if torch.cuda.is_available():
        images = images.cuda()
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test accuracy of the model on 10000 test images: %d %%' % (100 * correct / total))
torch.save(cnn.state_dict(), 'cnn.pkl')
