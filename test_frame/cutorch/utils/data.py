"""
Functions for loading and manipulating data
"""

from random import shuffle

import torch

global images, labels, batch, minibatch

class DataLoader:
    """ Prepares data to train/cross-validate/test. Eg: Batching, shuffling. """
    def __init__(self, data, batch_size, shuffled=False, cross_val=False):
        if isinstance(data, list):
            self.data = data
        else:
            raise TypeError("Dataset should be a list.")
        self.batch_size = batch_size
        self.to_shuffle = shuffled
        self.cross_val = cross_val

    def __getitem__(self, i):
        return self.mini_batches[i]

    def __iter__(self):
        self.i = 0
        if self.to_shuffle:
            #print("SHUFFLING ... ")
            self.shuffle_data()
        self.create_batches()
        if self.cross_val:
            self.num_batches -= 1
        return self

    def __next__(self):
        #print "Batch: " + str(self.i)
        if self.i < self.num_batches:
            minibatch =  self.mini_batches[self.i]
            self.i += 1
            return minibatch
        else:
            raise StopIteration
    
    def next(self):
        # Compatible with both python 2 & 3
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
        labels = torch.squeeze(labels, 1)  # To make this a vector
        self.num_batches = len(self.data) / self.batch_size
        self.mini_batches = []
        for batch in range(self.num_batches):
            b_start = batch * self.batch_size
            b_end = (batch + 1) * self.batch_size
            self.mini_batches.append((images[b_start:b_end], labels[b_start:b_end]))