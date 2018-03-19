""" 
Flip the given data (training)
Return appended (augmented dataset)
"""

import torch
import numpy as np


from data import dataset as dset

def flip(image=None):
    # Get data
    orig_dataset = dset.CIFAR10(directory='data', download=True, train=True)
    print(orig_dataset)
    orig_loader = dset.data_loader(orig_dataset.data, batch_size=dset.CIFAR10.train_size, shuffled=False)

    aug_dataset = augment_flipped(orig_loader)

def augment_flipped(data):
    i = 0
    for images, labels in data:
        for image, label in zip(images[:2], labels[:2]):
            t_image = image.numpy()
            t_image = np.flip(t_image, 2)  
            t_image = torch.from_numpy(t_image.copy()).type(torch.FloatTensor)
            print image, t_image

flip()
