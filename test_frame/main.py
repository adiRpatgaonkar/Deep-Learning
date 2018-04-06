from __future__ import print_function

import torch
import cutorch.nn as nn
import cutorch.nn.functionals as f

__dlevel__ = 0


class FCL(nn.Module):

    def __init__(self):
        super(FCL, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )

        self.layer1.see_modules()

    def forward(self, inputs):
        if __debug__:
            if __dlevel__ == 2:
                print("Input:{}".format(inputs))
                
        # Standardize
        inputs = f.standardize(inputs)

        # Fprop
        out = self.layer1(inputs)
        #out = self.layer2(out)

        if __debug__:
            if __dlevel__ == 1:
                print("Output:{}".format(out))
            if __dlevel__ == 2:
                pass
            if __dlevel__ == 3:
                print("Stdized input:{}".format(inputs))
            if __dlevel__ == 4:
                print(self.layer1['module', 0])
                print(self.layer1['parameters', 0])
        return out


def main():
    # Fully connected layer model
    fcl = FCL()
    criterion = nn.CrossEntropyLoss()

    # Input image
    image = torch.rand(1, 3, 32, 32)

    # Apply the n/w on the image
    outputs = fcl(image)
    loss = criterion(outputs, [1])
    print(loss)

if __name__ == '__main__':
    main()
