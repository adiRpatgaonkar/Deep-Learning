""" 
Flip the given data (training)
Return appended (augmented dataset)
"""
import sys
import torch
import numpy as np
import copy

from data import dataset as dset

class TransformData():

    def __init__(self, orig_dataset, transform, augment=False):
        self.transform = transform
        self.augment = augment
        if transform == 'flip':
            self.data = self.flip(orig_dataset, transform, augment)


    def flip(self, orig_dataset, transform, augment):
        data_aug = copy.deepcopy(orig_dataset)

        if augment:
            data_aug.data = orig_dataset.data[:]
        else:
            data_aug.data = []
        for image, label in orig_dataset.data:
            t_image = np.flip(image.numpy(), 2)  
            t_image = torch.from_numpy(t_image.copy()).type(torch.FloatTensor)
            data_aug.data.append((t_image, label))
        return data_aug.data