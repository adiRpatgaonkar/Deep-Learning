# System imports
from __future__ import print_function
import os
from termcolor import colored
import torch
import numpy as np
# Custom imports
import do_stuff as do
import Dataset as dset
import nnCustom as nnc

# Testing
def test(model, fitting_loader=None):
    """ Evaluate model results on test/train set """
    
    print("\n+++++++Testing+++++++\n")
    # Get data
    if fitting_loader is None:
        test_dataset = dset.CIFAR10(directory='data/', download=True, test=True)  
        test_loader = dset.data_loader(test_dataset.data, batch_size=dset.CIFAR10.test_size, shuffled=False)
    else:
        test_dataset = dset.CIFAR10(directory='data/', download=True, train=True)
        test_loader = fitting_loader
        
    for images, labels in test_loader:
        if do.args.GPU:
            images = images.cuda()
        model.test(images, labels)
    labels = torch.from_numpy(np.array(labels))
    print(colored('\n# Testing Loss:', 'red'), end="")
    print('[%.4f]' % model.loss)
    model.test_acc = model.optimum['TestAcc'] = \
        (torch.mean((model.predictions == labels).float()) * 100)  # Testing accuracy
    print(colored('\nTesting accuracy:', 'green'), end="")
    print(" = %.2f %%" % model.test_acc)
    
    if os.path.isfile('model_final.pkl'):
        t = nnc.load_model('model_final.pkl')
        print('Loss:', t['Loss'], '|', 'Testing accuracy:', t['TestAcc'], '%')
        if t['TestAcc'] < model.test_acc:
            print('\nThis is the best tested model.', end=" ")
        else:
            print('\nBetter models exist.')
    else:
        print('No tested models exist.')
    nnc.save_model('model_final.pkl', model)