from __future__ import print_function

import time

import cutorch
import cutorch.nn as nn
import cutorchvision.datasets as dsets
from cutorchvision.transforms import Transforms

__dlevel__ = 0

# Get train data for training and cross validation
train_dataset = dsets.CIFAR10(directory="cutorchvision/data",
                              download=True,
                              train=True)
# Data augmentation
train_dataset = Transforms(dataset=train_dataset,
                           lr_flip=True)


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


def main():
    # Fully connected layer model
    fcm = FCM()
    criterion = nn.CrossEntropyLoss()

    max_epochs = 100
    learning_rate = 0.05  # To be used with optimizer
    lr_decay = 0.00005
    optimizer = cutorch.optim.SGD(fcm.parameters,
                                       lr=learning_rate,
                                       max_epochs=max_epochs,
                                       lr_decay=lr_decay)
    time_start = time.time()
    for epoch in range(max_epochs):
        print("\nEpoch:[{}/{}]".format(epoch + 1, max_epochs))
        # Shuffle data before the training round
        train_loader = (cutorch.utils.data.DataLoader(
            data=train_dataset.data,
            batch_size=dsets.CIFAR10.batch_size,
            shuffled=True))
        for i, (images, ground_truths) in enumerate(train_loader):
            outputs = fcm(images)
            loss = criterion(outputs, ground_truths)
            loss.backward()
            optimizer.step()
            if (i + 1) % 50 == 0:
                print("Iter[{}/{}]".format(i + 1, len(train_loader)), end=" ")
                print("error:[{}]".format(loss.data))

    print("Time taken to train is {:.2f} seconds".format(time.time() - time_start))


if __name__ == "__main__":
    main()
