from __future__ import print_function

import os
import cPickle as pickle
from subprocess import call

saved_model_dir = 'outputs/models/'


def save(model, filename="model.pkl"):
    """ Save the status dictionary """
    if not os.path.exists(saved_model_dir):
        # print("Creating outputs/models/ directory")
        call("mkdir outputs && mkdir outputs/models", shell=True)
    elif os.path.exists(saved_model_dir+filename):
        # print("Removing previously saved model")
        call("rm -f " + saved_model_dir+filename, shell=True)
    print("\nSaving", end=" ")
    print(filename, 'to', saved_model_dir, end=" ... ")
    with open(saved_model_dir + filename, 'wb') as f:  # Overwrites any existing file.
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    print("done.")
    print('Model saved as %s' % saved_model_dir + filename)
    f.close()


def load(filename):
    """ Load model from existing pickle file """
    print('\nChecking saved models ...')
    print('\nLoading status dictionary from %s ... '
          % saved_model_dir + filename)
    # Get the saved log (status dictionary)
    if os.path.isfile(saved_model_dir + filename):
        model = pickle.load(open(saved_model_dir + filename, 'rb'))
    else:
        print("Model file not found.")
        return
    return model    
