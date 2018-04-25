from __future__ import print_function

import torch  # CUDA

import cutorch
import cutorch.nn as nn
import cutorch.nn.functionals as F
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms

if cutorch.gpu.available():
    print("\nGPU used: {}".format(torch.cuda.current_device()))
else:
    print("\nUsing CPU.")

# Global vars
global images, labels, outputs, predicted, loss
global train_loader, test_loader

# Hyperparameters
max_epochs = 100
learning_rate, lr_decay = 5e-2, 5e-7
reg = 1e-3

# For training
trainset = dsets.CIFAR10(dir="cutorchvision/data", download=True, 
                         train=True, form="tensor")
train_loader = cutorch.utils.data.DataLoader(data=trainset.data, 
                                             batch_size=100, 
                                             shuffled=True)

# For testing
testset = dsets.CIFAR10(dir="cutorchvision/data", download=True, 
                        test=True, form="tensor")
test_loader = cutorch.utils.data.DataLoader(data=testset.data, 
                                            batch_size=10000,
                                            shuffled=False)


# Fully connected layer model
class FCM(nn.Module):
    def __init__(self):
        super(FCM, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, 2048))

        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax())

        self.fc1.see_modules()
        self.fc2.see_modules()

    def forward(self, x):
        out = cutorch.standardize(x)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


fcm = FCM()
# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = cutorch.optim.Optimizer(fcm, lr=learning_rate, lr_decay=lr_decay, reg=reg)

fcm.train()  # Switch: training mode
for epoch in range(max_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if cutorch.gpu.used:
            images = images.cuda()
            labels = labels.cuda()   
        
        optimizer.zero_grad()
        outputs = fcm(images)
        loss = criterion(outputs, labels)
        loss.data += F.l2_reg(reg, fcm.parameters())
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print("\nL.R:[{:.5f}]".format(fcm.hypers('lr')), end=" ")
            print("Epoch:[{}/{}]".format(epoch + 1, max_epochs), end=" ")
            print("Iter:[{}/{}]".format(i + 1, train_loader.num_batches), end=" ")
            print("error:[{:.5f}]".format(loss.data), end=" ")
        if cutorch.gpu.used:  # GPU cache cleaning. Unsure of effectiveness
            torch.cuda.empty_cache()

fcm.eval()
total = 0
correct = 0
for images, labels in test_loader:
    if cutorch.gpu.used:
        images = images.cuda()  # Move image batch to GPU
        labels = labels.cuda()
    outputs = fcm(images)
    _, predicted = outputs.data
    total += len(labels)
    correct += (predicted == labels).sum()
    if cutorch.gpu.used:  # GPU cache cleaning. Unsure of effectiveness
        torch.cuda.empty_cache()
print("\nTest accuracy of the trained model on {} test images: {} %".format(total, 100*correct/total))

# Save final model
cutorch.save(fcm.state_dict(), "fcm_final.pkl")
