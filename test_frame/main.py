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
    		if __dlevel__ == 1: 
        		print("Output:{}".format(out))
        	if __dlevel__ == 2:
        		pass
    		if __dlevel__ == 3:
    			print("Stdized input:{}".format(inputs))
    		if __dlevel__ == 4:
        		print(self.layer1['module', 0])
        		print(self.layer1['parameters', 0])
        print("Output:{}".format(out))

def main():
	
	# Fully connected layer model
    fcl = FCL()
    
    # Input image
    image = torch.rand(1, 3, 32, 32)

    fcl(image)

if __name__ == '__main__':
    main()
