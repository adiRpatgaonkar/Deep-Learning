"""
Functions for loading and manipulating data
"""

from random import shuffle

import torch

# def DataLoader(data, batch_size, fit_testing=False, shuffled=False):
#     """ Prepares, shuffles given data x"""
#     if shuffled:
#         shuffle(data)
#     images, labels = [], []
#     for x, y in data:
#         images.append(x)
#         labels.append(y)
#     images = torch.stack(images, dim=0)
#     if fit_testing:
#         num_batches = 1
#     else:
#         num_batches = len(data) / batch_size
#     mini_batches = []
#     for batch in range(num_batches):
#         b_start = batch * batch_size
#         b_end = (batch + 1) * batch_size
#         mini_batches.append((images[b_start:b_end], labels[b_start:b_end]))
#     return mini_batches

global images, labels, batch, minibatch
class DataLoader:

    def __init__(self, data, batch_size, shuffled=False):
        if isinstance(data, list):
            self.data = data
        else:
            raise TypeError("Dataset should be a list.")
        self.batch_size = batch_size
        self.to_shuffle = shuffled

        # if self.to_shuffle:
        #     self.shuffle_data()

        # if not batch_size == len(data):
        #     self.create_batches()


    def __iter__(self):
        self.i = 0
        
        if self.to_shuffle:
            print("SHUFFLING ... ")
            self.shuffle_data()
        if not self.batch_size == len(self.data):
            self.create_batches()
        return self

    def __next__(self):
        print "Batch: " + str(self.i)
        if self.i < self.num_batches - 1:
            minibatch =  self.mini_batches[self.i]
            self.i += 1
            return minibatch
        else:
            raise StopIteration
    
    def next(self):
        return self.__next__()


    def shuffle_data(self):
        shuffle(self.data)

    def create_batches(self):
        images, labels = [], []
        for x, y in self.data:
            images.append(x)
            labels.append(y)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        self.num_batches = len(self.data) / self.batch_size
        self.mini_batches = []
        for batch in range(self.num_batches):
            b_start = batch * self.batch_size
            b_end = (batch + 1) * self.batch_size
            self.mini_batches.append((images[b_start:b_end], labels[b_start:b_end]))