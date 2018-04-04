""" Paramters class for the model """

class Parameter:

    def __init__(self, **kwargs):
		for key, value in kwargs.items():
			self.tag = key
			self.data = value
    
        
