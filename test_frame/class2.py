""" Test child class for base class in class1.py """

import class1 as nn

class layer1(nn.Module):
    
    def do_some(self):
        print 'Nothing'
 
       
    #def forward(self):
    #    return None



l_1 = layer1()
l_1.forward()


