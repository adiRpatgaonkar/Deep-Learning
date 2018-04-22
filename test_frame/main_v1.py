"""
# TODO
# PIPELINE
Correct abstractions in code
Class performance
# MODEL IMPROVEMENTS
Regularization
Momentum
LogSoftmax
"""
from __future__ import print_function

import cutorch.nn as nn
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms
from evaluate import *

if cutorch.gpu_check.available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    using_gpu = True
else:
    using_gpu = False

# Global vars
global images, ground_truths, outputs, loss
global train_loader, test_loader

# Hyperparameters
max_epochs = 10
learning_rate = 5e-2
lr_decay = 5e-5

# Get training data for training and Cross validation
train_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                              download=True, train=True,
                              form="tensor")
# Data augmentation
train_dataset = Transforms(dataset=train_dataset,
                           lr_flip=True)
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
optimizer = cutorch.optim.Optimizer(fcm, lr=learning_rate, lr_decay=lr_decay)

for epoch in range(max_epochs):
    train_loader = cutorch.utils.data.DataLoader(
        data=train_dataset.data, batch_size=100, shuffled=True)
    for i, batch in enumerate(train_loader[:-1]):
        images, ground_truths = batch
        if using_gpu:
            images = images.cuda()

        fcm.train()  # Switch: training mode
        optimizer.zero_grad()
        outputs = fcm(images)
        loss = criterion(outputs, ground_truths)
        loss.backward()
        optimizer.step()
        total, accuracy = evaluate(fcm, train_loader[-2:-1], "cross-validate")
        if (i + 1) % 50 == 0:
            print("\nL.R:[{:.5f}]".format(fcm.hypers('lr')), end=" ")
            print("Epoch:[{}/{}]".format(epoch + 1, max_epochs), end=" ")
            print("Iter:[{}/{}]".format(i + 1, len(train_loader)), end=" ")
            print("error:[{:.5f}]".format(loss.data), end=" ")
            print("CVal accuracy on {} images: {} %".format(total, accuracy))
        if using_gpu:  # GPU cache cleaning. Unsure of effectiveness
            torch.cuda.empty_cache()
    total = evaluate(fcm, test_loader, "test")
    print('\nTest accuracy on {} images: {} %'.format(total, fcm.results['accuracy']))
    optimizer.check_model(select=True)  # Save the best model

print("\nFinished training: {} examples for {} epochs.".format(len(train_dataset.data), max_epochs))

optimizer.check_model(store=True, name="fcm_test.pkl")
cutorch.save(fcm.state_dict(), "fcm.pkl")

# ++++ Load trained model & test it ++++ #
model = cutorch.load('fcm.pkl')
# model = cutorch.load('fcm_test_1.pkl')['best']
total = evaluate(model, test_loader, "test")
print('Test accuracy of the trained model on {} test images: {} %'.format(total, model.results['accuracy']))
