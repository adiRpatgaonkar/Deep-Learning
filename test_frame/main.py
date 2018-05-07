"""
# TODO
# PIPELINE
Correct abstractions in code
# MODEL IMPROVEMENTS
Momentum
LogSoftmax
"""
from __future__ import print_function

import time

import cutorch.nn as nn
import cutorch.nn.functionals as F
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms
from evaluate import *

if cutorch.gpu.available():
    print("\nGPU used: {}".format(torch.cuda.current_device()))
else:
    print("\nUsing CPU.")

# Global vars
global curr_time, time2train
global images, labels, outputs, predicted, loss
global train_loader, test_loader

# Hyperparameters
max_epochs = 100
learning_rate, lr_decay = 5e-2, 5e-7
reg = 1e-3

# Get training data for training and Cross validation
trainset = dsets.CIFAR10(dir="cutorchvision/data", download=True, 
                         train=True, form="tensor")
# Data augmentation
trainset = Transforms(dataset=trainset, lr_flip=True, crop=False)
train_loader = cutorch.utils.data.DataLoader(data=trainset.data, 
                                             batch_size=100, 
                                             shuffled=True, 
                                             cross_val=True)

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
            nn.Linear(32 * 32 * 3, 2048),
            nn.ReLU())

        self.fc2 = nn.Sequential(
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

time_start = time.time()
for epoch in range(max_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if cutorch.gpu.used:
            images = images.cuda()
            labels = labels.cuda()   

        fcm.train()  # Switch: training mode
        optimizer.zero_grad()
        curr_time = time.time()  # Time 2 train a batch in train set
        outputs = fcm(images)
        loss = criterion(outputs, labels)
        loss.data += F.l2_reg(reg, fcm.parameters())
        loss.backward()
        optimizer.step()
        time2train = cutorch.utils.time_log(time.time() - curr_time)
        total, accuracy = evaluate(fcm, train_loader[-1], "cross-validate")
        if (i + 1) % 100 == 0:
            print("\nL.R:[{:.5f}]".format(fcm.hypers('lr')), end=" ")
            print("Epoch:[{}/{}]".format(epoch + 1, max_epochs), end=" ")
            print("Iter:[{}/{}]".format(i + 1, train_loader.num_batches), end=" ")
            print("error:[{:.5f}]".format(loss.data), end=" ")
            print("Time:[{}]".format(time2train))
            print("CVal accuracy on {} images: {} %".format(total, accuracy))
        if cutorch.gpu.used:  # GPU cache cleaning. Unsure of effectiveness
            torch.cuda.empty_cache()
    total = evaluate(fcm, test_loader, "test")
    print('\nTest accuracy on {} images: {} %'.format(total, fcm.results['accuracy']))
    optimizer.check_model(select=True)  # Keep a snapshot of the best model

net_time = cutorch.utils.time_log(time.time() - time_start)
print("\nFinished training: {} examples for {} epochs.".format(len(trainset.data), max_epochs))
print("Time[training + cross-validation + testing + best model selection]: {}".format(net_time))

# Evaluate best model found while training
total = evaluate(optimizer.state['model'], test_loader, "test")
print('Test accuracy of the trained model on {} test images: {} %'.format(total,
                                                                          optimizer.state['model'].results['accuracy']))

# Save best trained model
optimizer.check_model(store=True, name="fcm_best")
# Save final model
cutorch.save(fcm.state_dict(), "fcm_final")

# ++++ Load trained model & test it ++++ #
# model = cutorch.load('fcm_final.pkl') # Final model
# model = cutorch.load('fcm_best.pkl')['model']  # Best trained model
# total = evaluate(model, test_loader, "test")
# print('Test accuracy of the trained model on {} test images: {} %'.format(total, model.results['accuracy']))
