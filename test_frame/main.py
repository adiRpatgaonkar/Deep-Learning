from __future__ import print_function

import torch
import cutorch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1541),
            nn.ReLU(),
            nn.Linear(1541, 10),
            nn.ReLU()
        )

        self.l1.see_modules()

    def forward(self, inputs):
        out = self.l1.forward(inputs)
        if __debug__:
            print(out)


def main():
    cnn = CNN()

    image = torch.rand(1, 3072)
    cnn.forward(image)

    if __debug__:
        for module, param in cnn.l1.parameters().items():
            print('Module:', module)
            print('Parameters:', param)
            for p in param:
                print(p.tag, p.data)


if __name__ == '__main__':
    main()
