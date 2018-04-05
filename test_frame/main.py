from __future__ import print_function

import torch
import cutorch.nn as nn
import cutorch.nn.functionals as f

__dlevel__ = 1

class FCL(nn.Module):
    def __init__(self):
        super(FCL, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(32*32*3, 1541),
            nn.ReLU(),
            nn.Linear(1541, 10)
        )

        self.layer1.see_modules()

    def forward(self, inputs):
    	if __debug__ and __dlevel__ == 2:
    		print("Input:{}".format(inputs))

       	inputs = f.standardize(inputs)
        
        # Fprop
        out = self.layer1(inputs)

    	if __debug__:
    		if __dlevel__ == 3:
    			print("Stdized input:{}".format(inputs))
    		if __dlevel__ == 1: 
        		print("Output:{}".format(outs))


def main():
    fcl = FCL()

    image = torch.rand(10000, 3, 32, 32)

    fcl.forward(image)

    if __debug__: # Check module(s) parameters
    	if __dlevel__ == 4:
        for module, param in fcl.l1.parameters().items():
            print('Module:', module)
            print('Parameters:', param)
            for p in param:
                print(p.tag, p.data)


if __name__ == '__main__':
    main()
