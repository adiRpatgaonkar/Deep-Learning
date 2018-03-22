""" 
Flip the given data (training)
Return appended (augmented dataset)
"""

# System imports
from __future__ import print_function
import torch
import numpy as np
from matplotlib.pyplot import imshow, show
from data.dataset import CIFAR10

# Global variables
global image, ground_truth, np_image


class Transforms:
    """
    Data transforms (for data augmentation)

    """
    def __init__(self, dataset, lr_flip=False, ud_flip=False, crop=False, rotate90=False, times=None):
        """ Transform data acc. to the transform param """

        # Alias of <dataset> object
        self.original_dataset = dataset
        # Augmented data
        self.data = []
        # Transforms
        self.transform_data(lr_flip, ud_flip, crop, rotate90, times)

    def transform_data(self, lr_flip, ud_flip, crop, rotate90, times):
        """ Transform the data """

        if not lr_flip and not ud_flip and not crop and not rotate90:
            print("No transforms done.")
            return

        print("Augmenting data:")
        if lr_flip:
            print("Flipping training examples horizontally ...", end=" ")
            for image, ground_truth in self.original_dataset.data:
                np_image = np.flip(image.numpy(), 2)
                np_image = torch.from_numpy(np_image.copy()).type(torch.FloatTensor)
                self.data.append((np_image, ground_truth))
            print("done.")
        if ud_flip:
            print("Flipping training examples upside down ...", end=" ")
            for image, ground_truth in self.original_dataset.data:
                np_image = np.flip(image.numpy(), 2)
                np_image = torch.from_numpy(np_image.copy()).type(torch.FloatTensor)
                self.data.append((np_image, ground_truth))
            print("done.")
        if crop:
            print("Cropping training examples ...", end=" ")
            for image, ground_truth in self.original_dataset.data:
                image[:, 0] = image[0, :] = image[:, -1] = image[-1, :] = 0
                self.data.append((image, ground_truth))
            print("done.")
        if rotate90:
            if times is None:
                print("No rotation.")
            else:
                print("Rotating images by %d degrees ..." % 90 * times, end=" ")
                for image, ground_truth in self.original_dataset.data:
                    np_image = np.rot90(image.numpy(), k=1, axes=(1, 2))
                    np_image = torch.from_numpy(np_image.copy()).type(torch.FloatTensor)
                    self.data.append((np_image, ground_truth))
                print("done.")
        
        self.data += self.original_dataset.data

        return


def see(image):
    """ Use the vision """
    image = image.cpu()
    image = image.numpy().reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
    imshow(image)
    show()
