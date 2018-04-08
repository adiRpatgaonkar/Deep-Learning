from __future__ import print_function

import cutorch
import cutorch.nn as nn
import cutorchvision.datasets as dsets

__dlevel__ = 0

# Get train data for training and cross validation
train_dataset = dsets.CIFAR10(directory='cutorchvision/data',
                              download=True,
                              train=True)

train_loader = (
    cutorch.utils.data.DataLoader(data=train_dataset.data,
                                  batch_size=1,
                                  shuffled=True))


class FCM(nn.Module):

    def __init__(self):
        super(FCM, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
            nn.Softmax()
        )
        self.inputs = 0
        self.output = 0
        self.fc.see_modules()

    def forward(self, inputs):
        if __debug__:
            if __dlevel__ == 2:
                print("Input:{}".format(inputs))
        self.inputs = inputs
        # Standardize
        self.inputs = cutorch.standardize(self.inputs)
        # Fprop
        out = self.fc(self.inputs)
        # out = self.layer2(out)
        # +++++++++++ Debug +++++++++++ #
        if __debug__:
            if 2 >= __dlevel__ >= 1:
                print("Output:{}\n".format(out))
                print("Modules:{}\n".format(self.fc._modules))
                print("Forward-Path:{}\n".format(self.fc._forward_hooks))
                print("Backward-Path:{}\n".format(self.fc._backward_hooks))
            if __dlevel__ == 3:
                print("Stdized input:{}".format(self.inputs))
            if __dlevel__ == 4:
                print(self.fc['module', 0])
                print(self.fc['parameters', 0])
        # +++++++++++ Debug +++++++++++ #
        self.output = out
        return out

    def backward(self, targets):
        self.fc.backward(targets)


def main():
    # Fully connected layer model
    model = FCM()
    criterion = nn.CrossEntropyLoss()
    epochs = 5
    for epoch in range(epochs):
        print("Epoch:", epoch)
        for images, ground_truths in train_loader[0:1]:
            # Apply the n/w on the image
            outputs = model(images)
            loss = criterion(outputs, ground_truths)
            print(loss, loss.data)
            model.backward(ground_truths)


if __name__ == '__main__':
    main()
