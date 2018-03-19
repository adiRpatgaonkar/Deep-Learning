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


def test(model, fitting_loader=None):
    """ Evaluate model results on test/train set """
    global images, ground_truths
    args = arguments()

    print("\n+++++++     TESTING     +++++++\n")
    model.show_log(test=True)
    # Get data
    test_dataset = dset.CIFAR10(directory='data', download=True, test=True)
    if fitting_loader is None:
        test_loader = dset.data_loader(test_dataset.data, batch_size=dset.CIFAR10.test_size, shuffled=False)
    else:
        test_loader = fitting_loader
        
    for images, ground_truths in test_loader:
        if using_gpu():
            images = images.cuda()
        model.test(images, ground_truths)
        # Clear cache if using GPU (Unsure of effectiveness)
        if using_gpu():
                torch.cuda.empty_cache()
    ground_truths = torch.from_numpy(np.array(ground_truths))
    print(colored('\n# Testing Loss:', 'red'), end="")
    print('[%.4f]' % model.loss)
    model.test_acc = model.optimum['TestAcc'] = \
        (torch.mean((model.predictions == ground_truths).float()) * 100)  # Testing accuracy
    print(colored('\nTesting accuracy:', 'green'), end="")
    print(" = %.2f %%" % model.test_acc)

    # Tested model status
    if args.TRAIN:
        model.model_tested = model.optimum['Tested'] = True    
    print("\nModel status (current):")
    print("{ Fitting tested:", model.optimum['Fitting tested'], "|", "Trained:", model.optimum['Trained'], "|", 
          "Tested:", model.optimum['Tested'], "|", "Inferenced:", model.optimum['Inferenced'], "}")
    print("{ Loss:", model.optimum['Loss'], "||", model.optimum['TestAcc'], "% }\n")

    model.set_logs()
    # Saving fitted model    
    if args.SAVE:
        save_model(args.SAVE, model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            save_model('model.pkl', model)
        else:
            print('Not saving model.')
