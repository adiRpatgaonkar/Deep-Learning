"""
Functions for loading and manipulating data
"""

from random import shuffle

import torch

def DataLoader(data, batch_size, fit_testing=False, shuffled=False):
    """ Prepares, shuffles given data x"""
    if shuffled:
        shuffle(data)
    images, labels = [], []
    for x, y in data:
        images.append(x)
        labels.append(y)
    images = torch.stack(images, dim=0)
    if fit_testing:
        num_batches = 1
    else:
        num_batches = len(data) / batch_size
    mini_batches = []
    for batch in range(num_batches):
        b_start = batch * batch_size
        b_end = (batch + 1) * batch_size
        mini_batches.append((images[b_start:b_end], labels[b_start:b_end]))
    return mini_batches
