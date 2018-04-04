from __future__ import print_function

import torch
import cutorch.nn as nn
import cutorch.nn.functionals as f


class FCL(nn.Module):
    def __init__(self):
        super(FCL, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1541),
            nn.ReLU(),
            nn.Linear(1541, 10),
            nn.ReLU()
        )

        self.l1.see_modules()

    def forward(self, inputs):
        inputs = f.standardize(inputs)
        out = self.l1.forward(inputs)
        print(out)


def main():
    fcl = FCL()

    image = torch.rand(10000, 3, 32, 32)

    fcl.forward(image)

    if __debug__: # Check module(s) parameters
        for module, param in fcl.l1.parameters().items():
            print('Module:', module)
            print('Parameters:', param)
            for p in param:
                print(p.tag, p.data)


if __name__ == '__main__':
    main()
