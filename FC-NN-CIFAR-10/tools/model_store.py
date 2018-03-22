""" Save/load model """
# System imports
from __future__ import print_function

import os
import sys
import pickle
from subprocess import call

# Custom imports
from tools.create import create_model

# Some global vars
global saved_model_dir
saved_model_dir = 'outputs/models/'


def save_model(filename="model.pkl", model=None):
    """ Save the status dictionary """
    if not model:
        print("No model found")
    if not os.path.exists(saved_model_dir):
        print("Creating outputs/models/ directory")
        call("mkdir outputs && mkdir outputs/models", shell=True)
    print('\nSaving', end=" ")
    print(filename, 'to', saved_model_dir, end=" ... ")
    f = open(saved_model_dir + filename, 'wb')
    pickle.dump(model.optimum, f)
    print("done.")
    print('Model saved as %s' % saved_model_dir + filename)
    f.close()


def load_model(filename):
    """ Load model dictionary and rebuild the model """
    print('\nChecking saved models ...')
    print('\nLoading status dictionary from %s ... '
          % saved_model_dir + filename)
    # Get the saved log (status dictionary)
    if os.path.isfile(saved_model_dir + filename):
        t = pickle.load(open(saved_model_dir + filename, 'rb'))
    else:
        print("Model file not found.")
        sys.exit(1)

    # Create a model
    model = create_model()
    # Give the status dictionary to the created net
    model.optimum = t
    # Use the given status dictionary to get 
    # the model up on its feet
    model.get_logs()

    return model
