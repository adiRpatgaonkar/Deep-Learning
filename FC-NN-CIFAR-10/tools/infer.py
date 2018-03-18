""" Inference code for saved models """

# System imports
from __future__ import print_function

import numpy as np
import torch
from matplotlib.pyplot import ylabel, imshow, show, xlabel

# Custom imports
from data import dataset as dset

from libs.check_args import arguments, using_gpu

from tools.model_store import save_model


def inferences(model, fitting_loader=None, all_exp=False):
    """ Display model results i.e. predictions on test/train set """
    args = arguments()

    global images, ground_truths
    print("\n+++++++     INFERENCE     +++++++\n")
    model.show_log(infer=True)
    # Get data
    test_dataset = dset.CIFAR10(directory='data', download=True, test=True)
    if fitting_loader is None:
        infer_loader = dset.data_loader(test_dataset.data, batch_size=dset.CIFAR10.test_size, shuffled=False)
    else:
        infer_loader = fitting_loader
    
    print("Test accuracy:", model.optimum['TestAcc'], '%')    
    for images, ground_truths in infer_loader:
        if using_gpu:
            images = images.cuda()
        model.test(images, ground_truths)
        ground_truths = torch.from_numpy(np.array(ground_truths))
    if all_exp:
        for example in range(dset.CIFAR10.test_size):
            print("Ground truth: (%d) %s || Predicition: (%d) %s || Confidence: %.2f %" %
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
                break
            print('Ground truth: (%d) %s' % (int(ground_truths[example]),
                                             dset.CIFAR10.classes[int(ground_truths[example])]))
            imshow(images[example])
            xlabel(str(int(model.predictions[example])) + ' : ' +
                   dset.CIFAR10.classes[int(model.predictions[example])])
            ylabel('Confidence: ' + str(format(model.output[-1][example] * 100, '.2f')) + '%')
            show()
    
    model.set_logs()        
    # Model status
    model.model_infered = model.optimum['Inferenced'] = True    
    print("\nModel status (current):")
    print("{ Fitting tested:", model.optimum['Fitting tested'], "|", "Trained:", model.optimum['Trained'], "|", 
          "Tested:", model.optimum['Tested'], "|", "Inferenced:", model.optimum['Inferenced'])
    print("{ Loss:", model.optimum['Loss'], "||", model.optimum['TestAcc'], "% }\n")
    
    # Saving inferenced model    
    if args.SAVE:
        save_model('model.pkl', model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            save_model('model.pkl', model)
        else:
            print('Not saving model.')
