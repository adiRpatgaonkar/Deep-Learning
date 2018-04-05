""" Parameters class for the model """


class Parameter:

    def __init__(self, **kwargs):
    	print type(kwargs)
        for key, value in kwargs.items():
            self.tag = key
            self.data = value
