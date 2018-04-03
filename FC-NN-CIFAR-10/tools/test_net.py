""" Testing code for saved models """

# System imports
from __future__ import print_function

import numpy as np
import torch
from termcolor import colored

# Custom imports
from libs.check_args import arguments, using_gpu
from tools.model_store import save_model
from data import dataset as dset


def test(model, data_loader=None):
    """ Evaluate model results on test/train set """
    global images, ground_truths
    args = arguments()

    print("\n+++++++     TESTING     +++++++\n")
    model.show_log(test=True)

    # Get data
    test_dataset = dset.CIFAR10(directory='data',
                                download=True,
                                test=True)

    # If fitting is done, get 
    # the correct dataset to be tested
    if data_loader is None:
        test_loader = dset.data_loader(data=test_dataset.data,
                                       batch_size=dset.CIFAR10.test_size,
                                       shuffled=False)
    else:
        test_loader = data_loader

    # In case test set is divided in batches    
    for images, ground_truths in test_loader:
        if using_gpu():
            images = images.cuda()
        model.test(images, ground_truths)
        # Clear cache if using GPU (Unsure of effectiveness)
        if using_gpu():
            torch.cuda.empty_cache()

    # Convert tensor --> numpy ndarray
    ground_truths = torch.from_numpy(np.array(ground_truths))

    # Print testing loss & accuracy
    print(colored('\n# Testing Loss:', 'red'), end="")
    print('[%.4f]' % model.test_loss)
    model.test_acc = model.optimum['TestAcc'] = \
        (torch.mean((model.predictions == ground_truths).float()) * 100)  # Testing accuracy
    print(colored('\nTesting accuracy:', 'green'), end="")
    print(" = %.2f %%" % model.test_acc)

    # Tested model status
    if args.TRAIN:
        model.tested = True

    model.show_log(curr_status=True)
    model.save_state()

    # Saving fitted model    
    if args.SAVE:
        save_model(args.SAVE, model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            save_model('model.pkl', model)
        else:
            print('Not saving model.')
