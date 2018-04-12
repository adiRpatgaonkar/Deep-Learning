from __future__ import print_function

import time

import torch # CUDA #
import cutorch
import cutorch.nn as nn
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms

__dlevel__ = 0

if cutorch.gpu_check.available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    using_gpu = True
else:
    using_gpu = False

# Get train data for training and cross validation
train_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                              download=True,
                              train=True,
                              form="tensor")
# Data augmentation
train_dataset = Transforms(dataset=train_dataset,
                           lr_flip=True,
                           crop=False)

test_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                                download=True,
                                test=True,
                                form="tensor")

test_loader = cutorch.utils.data.DataLoader(data=test_dataset.data,
                               batch_size=10000,
                               shuffled=False)


class FCM(nn.Module):
    def __init__(self):
        super(FCM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Softmax())

        self.fc.see_modules()

    def forward(self, x):
        out = cutorch.standardize(x)
        out = self.fc(out)
        return out



# Fully connected layer model
fcm = FCM()
criterion = nn.CrossEntropyLoss()

# Hyperparameters
max_epochs = 1
learning_rate = 5e-2  
lr_decay = 5e-5
reg = 1e-3
optimizer = cutorch.optim.SGD(fcm,
                              lr=learning_rate,
                              lr_decay=lr_decay,
                              reg_strength=reg)
# Training mode
fcm.train()
time_start = time.time()
for epoch in range(max_epochs):
    print("\nEpoch:[{}/{}]".format(epoch + 1, max_epochs))
    train_loader = (cutorch.utils.data.DataLoader(
        data=train_dataset.data,
        batch_size=100,
        shuffled=True))
    for i, batch in enumerate(train_loader):
        images, ground_truths = batch
        if using_gpu:
            images = images.cuda() # Move image batch to GPU
        outputs = fcm(images)
        loss = criterion(outputs, ground_truths)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            # Optional
            #print("L.R:[{:.5f}]".format(fcm._hyperparameters['lr']), end=" ")
            print("Iter:[{}/{}]".format(i + 1, len(train_loader)), end=" ")
            print("error:[{}]".format(loss.data))
        if using_gpu:
            torch.cuda.empty_cache()
time2train = time.time() - time_start
time2train = cutorch.utils.time_log(time2train)

print("\nFinished training:\n{} examples, {} epochs: in {}".format(len(train_dataset.data), 
                                                                    max_epochs, time2train))
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
print('Test accuracy of the model on {} test images: {} %'.format(int(total), 100*(correct/total)))
