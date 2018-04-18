"""
TODO
# PIPELINE
Correct abstractions in code
Best model saved
Test on best parameters
Class performance
# MODEL IMPROVEMENTS
Regularization
Momentum
LogSoftmax
"""
from __future__ import print_function

import time

import torch  # CUDA #
import cutorch
import cutorch.nn as nn
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms


if cutorch.gpu_check.available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    using_gpu = True
else:
    using_gpu = False

# Hyperparameters
max_epochs = 10
learning_rate = 5e-2
lr_decay = 5e-5

# Get train data for training and TODO: Cross validation
train_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                              download=True, train=True,
                              form="tensor")
# Data augmentation
train_dataset = Transforms(dataset=train_dataset,
                           lr_flip=True, crop=False)

test_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                             download=True, test=True,
                             form="tensor")

test_loader = cutorch.utils.data.DataLoader(data=test_dataset.data,
                                            batch_size=10000, 
                                            shuffled=False)

# Fully connected layer model
class FCM(nn.Module):
    def __init__(self):
        super(FCM, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 10),
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
optimizer = cutorch.optim.SGD(fcm, lr=learning_rate, lr_decay=lr_decay)

# Training mode
time_start = time.time()
for epoch in range(max_epochs):
    print("\nEpoch:[{}/{}]".format(epoch+1, max_epochs))
    train_loader = cutorch.utils.data.DataLoader(
        data=train_dataset.data, batch_size=100, shuffled=True)
    for i, batch in enumerate(train_loader):
        images, ground_truths = batch
        if using_gpu:
            images = images.cuda()  # Move image batch to GPU
        fcm.train()
        time_train = time.time() # Time 2 train a batch
        outputs = fcm(images)
        loss = criterion(outputs, ground_truths)
        loss.backward()
        optimizer.step()
        time2train = cutorch.utils.time_log(time.time() - time_train)
        if (i+1) % 50 == 0:
            print("L.R:[{:.5f}]".format(fcm.hypers('lr')), end=" ")
            print("Iter:[{}/{}]".format(i+1, len(train_loader)), end=" ")
            print("error:[{}]".format(loss.data), end=" ")
            print("Time:[{}]".format(time2train))
        if using_gpu: # GPU cache cleaning. Unsure of effectiveness
            torch.cuda.empty_cache()
net_time = cutorch.utils.time_log(time.time() - time_start)
print("\nFinished training:\n{} examples, {} epochs: in {}.".format(len(train_dataset.data),
                                                                   max_epochs, net_time))
# Change to evaluation mode
fcm.eval()
correct = 0.0
total = 0
for images, labels in test_loader:
    if using_gpu:
        images = images.cuda()
    labels = torch.LongTensor(labels)
    outputs = fcm(images)
    _, predicted = outputs.data
    total += len(labels)
    correct += (predicted.cpu() == labels).sum()
print('Test accuracy of the model on {} test images: {} %'.format(int(total), 100 * (correct / total)))
