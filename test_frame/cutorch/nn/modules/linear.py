""" Linear layer class """


from .module import Module
from .. import Paramater


class Linear(Module):
    """Linear Layer class"""

    LayerName = 'Linear'

    def __init__(self, in_features, out_features):
        # print 'Linear layer created'
        # allocate size for the state variables appropriately
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.rand(in_features, out_features))
        self.bias = torch.zeros(1, out_features).type(default_tensor_type())


    
