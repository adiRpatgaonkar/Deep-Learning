""" NN module class """


class Module(object):
    """ Base class for all nn modules """
    def __init__(self):
        self.x = 0

    def forward(self, *input):
        """
        Should be overriden by every module
        """


    def add(self)


    def parameters(self, **kwargs):
