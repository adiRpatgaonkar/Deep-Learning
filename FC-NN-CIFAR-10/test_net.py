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
def test(model, fitting_loader=None, fitting=False):
    """ Evaluate model results on test/train set """
    
    print("\n+++++++     TESTING     +++++++\n")
    # Get data
    if fitting_loader is None:
        test_dataset = dset.CIFAR10(directory='data', download=True, test=True)  
        test_loader = dset.data_loader(test_dataset.data, batch_size=dset.CIFAR10.test_size, shuffled=False)
    else:
        test_dataset = dset.CIFAR10(directory='data', download=True, train=True)
        test_loader = fitting_loader
        
    for images, labels in test_loader:
        if do.use_gpu:
            images = images.cuda()
        model.test(images, labels)
    labels = torch.from_numpy(np.array(labels))
    print(colored('\n# Testing Loss:', 'red'), end="")
    print('[%.4f]' % model.loss)
    model.test_acc = model.optimum['TestAcc'] = \
        (torch.mean((model.predictions == labels).float()) * 100)  # Testing accuracy
    print(colored('\nTesting accuracy:', 'green'), end="")
    print(" = %.2f %%" % model.test_acc)

    # Tested model status
    if do.args.TRAIN:
        model.model_tested = model.optimum['Tested'] = True    
    print("\nModel status (current):")
    print("{ Fitting tested:", model.optimum['Fitting tested'], "|", "Trained:", model.optimum['Trained'], "|", 
          "Tested:", model.optimum['Tested'], "|", "Inferenced:", model.optimum['Inferenced'], "}")
    print("{ Loss:", model.optimum['Loss'], "||", model.optimum['TestAcc'], "% }\n")
    
    # Comparison with saved models
#    if not do.args.LOAD:
#        if os.path.isfile('model.pkl'):
#            t = nnc.load_model('model.pkl')
#            print("\nModel status (previously saved):")
#            print("{ Fitting tested:", t['Fitting tested'], "|", "Trained:", t['Trained'], "|", 
#              "Tested:", t['Tested'], "|", "Inferenced:", t['Inferenced'], "}")
#            print("{ Loss:", t['Loss'], "||", "Testing accuracy:", t['TestAcc'], "% }\n")
#            if t['TestAcc'] < model.test_acc:
#                print("\nThis is the best tested model.", end=" ")
#            else:
#                print("\nBetter models exist.")
#        else:
#            print("No tested models exist.")

    model.set_logs()
    # Saving tested model
    if do.args.SAVE:
        nnc.save_model('model.pkl', model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            nnc.save_model('model.pkl', model)
        else:
            print('Not saving model.')