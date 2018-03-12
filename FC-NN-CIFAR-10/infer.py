"""Inference code for saved models"""

from __future__ import print_function
from matplotlib.pyplot import ylabel, imshow, plot, show, xlabel
import numpy as np
import torch

import nnCustom as nnc
import do_stuff as do
import Dataset as dset


# Inference the model
def inferences(model, fitting_loader=None, all_exp=False):
    """ Display model results i.e. predictions on test/train set """
    
    print("\n+++++++Inference+++++++\n")
    # Get data
    if fitting_loader is None:
        test_dataset = dset.CIFAR10(directory='data/', download=True, test=True)
        infer_loader = dset.data_loader(test_dataset.data, batch_size=dset.CIFAR10.test_size, shuffled=False)
    else:
        test_dataset = dset.CIFAR10(directory='data/', download=True, train=True)
        infer_loader = fitting_loader
        
    for images, ground_truths in infer_loader:
            if do.args.GPU:
                images = images.cuda()
            model.test(images, ground_truths)
            ground_truths = torch.from_numpy(np.array(ground_truths))
    if all_exp:
        for example in range(dset.CIFAR10.test_size):
            print('Ground truth: (%d) %s || Predecition: (%d) %s || Confidence: %.2f %' %
                  (ground_truths[example], dset.CIFAR10.classes[int(ground_truths[example])],
                   int(model.predictions[example]),
                   dset.CIFAR10.classes[int(model.predictions[example])],
                   model.output[-1][example] * 100))
    else:
        images = images.cpu()
        images = \
            (images.numpy().reshape(dset.CIFAR10.test_size, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8"))
        while True:
            example = input("Which test example? (0-9999): ")
            if example < 0 or example >= dset.CIFAR10.test_size:
                return
            print('Ground truth: (%d) %s' % (int(ground_truths[example]),
                                             dset.CIFAR10.classes[int(ground_truths[example])]))
            imshow(images[example])
            xlabel(str(int(model.predictions[example])) + ' : ' +
                   dset.CIFAR10.classes[int(model.predictions[example])])
            ylabel('Confidence: ' + str(format(model.output[-1][example] * 100, '.2f')) + '%')
            show()
