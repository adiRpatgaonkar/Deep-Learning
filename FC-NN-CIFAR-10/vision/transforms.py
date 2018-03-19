""" 
Flip the given data (training)
Return appended (augmented dataset)
"""
from __future__ import print_function
import sys
import torch
import numpy as np
import copy

from data import dataset as dset

class TransformData():

    def __init__(self, orig_dataset, transform):
        self.transform = transform
        if transform == 'flip':
            self.data = self.flip(orig_dataset)
        if transform == 'crop':
            self.data = self.crop(orig_dataset)

    def flip(self, orig_dataset):
        data_aug = copy.deepcopy(orig_dataset)
        data_aug.data = orig_dataset.data[:]
        print("Horizontally flipping training examples ...", end = " ")
        for image, label in orig_dataset.data:
            t_image = np.flip(image.numpy(), 2)  
            t_image = torch.from_numpy(t_image.copy()).type(torch.FloatTensor)
            data_aug.data.append((t_image, label))
        print("done.")
        return data_aug.data

    def crop(self, orig_dataset):
        data_aug = copy.deepcopy(orig_dataset)
        data_aug.data = orig_dataset.data[:]
        print("Cropping training examples ...", end = " ")
        for image, label in orig_dataset.data:
            image[:, 0] = image[0, :] = image[:, -1] = image[-1, :] = 0  
            data_aug.data.append((image, label))
        print("done.")
        return data_aug.data