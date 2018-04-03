""" Test OOP Python """

class Module(object):

    def __init__(self):
        self.x = 0
        self.y = 0
        self.out = 0

    def forward(self):
        self.out = self.x + 5
        self.out *= (self.y + 7)
        print(self.out)

