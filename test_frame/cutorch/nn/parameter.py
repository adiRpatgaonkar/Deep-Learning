""" Parameters class for the model """


class Parameter:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'require_gradient':
                if isinstance(value, bool):
                    self.req_gradient = value
                continue
            self.tag = key
            self.data = value
