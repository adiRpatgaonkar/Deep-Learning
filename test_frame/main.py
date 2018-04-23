"""
# TODO
# PIPELINE
Correct abstractions in code
Class performance
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

if cutorch.gpu_check.available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    using_gpu = True
else:
    using_gpu = False

# Global vars
global curr_time, time2train
global images, ground_truths, outputs, predicted, loss
global train_loader, test_loader

# Hyperparameters
max_epochs = 50
learning_rate, lr_decay = 5e-2, 5e-5
reg = 1e-3

# Get training data for training and Cross validation
train_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                              download=True, train=True,
                              form="tensor")
# Data augmentation
train_dataset = Transforms(dataset=train_dataset,
                           lr_flip=True, crop=False)
                           
# For testing
test_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                             download=True, test=True,
                             form="tensor")
# Testing data for validation
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
optimizer = cutorch.optim.Optimizer(fcm, lr=learning_rate, lr_decay=lr_decay, reg=reg)

time_start = time.time()
for epoch in range(max_epochs):
    train_loader = cutorch.utils.data.DataLoader(
        data=train_dataset.data, batch_size=100, shuffled=True)
    for i, batch in enumerate(train_loader[:-1]):
        images, ground_truths = batch
        if using_gpu:
            images = images.cuda()

        fcm.train()  # Switch: training mode
        optimizer.zero_grad()
        curr_time = time.time()  # Time 2 train a batch in train set
        outputs = fcm(images)
        loss = criterion(outputs, ground_truths)
        loss.data += F.l2_reg(reg, fcm.parameters())
        loss.backward()
        optimizer.step()
        time2train = cutorch.utils.time_log(time.time() - curr_time)
        total, accuracy = evaluate(fcm, train_loader[-2:-1], "cross-validate")
        if (i + 1) % 50 == 0:
            print("\nL.R:[{:.5f}]".format(fcm.hypers('lr')), end=" ")
            print("Epoch:[{}/{}]".format(epoch + 1, max_epochs), end=" ")
            print("Iter:[{}/{}]".format(i + 1, len(train_loader)), end=" ")
            print("error:[{:.5f}]".format(loss.data), end=" ")
            print("Time:[{}]".format(time2train))
            print("CVal accuracy on {} images: {} %".format(total, accuracy))
        if using_gpu:  # GPU cache cleaning. Unsure of effectiveness
            torch.cuda.empty_cache()
    total = evaluate(fcm, test_loader, "test")
    print('\nTest accuracy on {} images: {} %'.format(total, fcm.results['accuracy']))
    optimizer.check_model(select=True) # Keep a snapshot of the best model

net_time = cutorch.utils.time_log(time.time() - time_start)
print("\nFinished training: {} examples for {} epochs.".format(len(train_dataset.data), max_epochs))
print("Time[training + cross-validation + testing + best model]: {}".format(net_time))

# Evaluate best model found while training
# total = evaluate(optimizer.state['model'], test_loader, "test")
# print('Test accuracy of the trained model on {} test images: {} %'.format(total, optimizer.state['model'].results['accuracy']))

# Save best trained model
optimizer.check_model(store=True, name="fcm_best")
# Save final model
cutorch.save(fcm.state_dict(), "fcm_final")

# # ++++ Load trained model & test it ++++ #
# model = cutorch.load('fcm.pkl') # Final model
# # model = cutorch.load('fcm_test_1.pkl')['model']  # Best trained model
# total = evaluate(model, test_loader, "test")
# print('Test accuracy of the trained model on {} test images: {} %'.format(total, model.results['accuracy']))
