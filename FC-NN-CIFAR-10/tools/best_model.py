""" Search, compare and find the best model """

# System imports
from __future__ import print_function
import os
import pickle
from subprocess import call
import operator as op


def best_model_selection(replace=False):
    # Change the dir to appropriate one.
    os.chdir("outputs/models/")

    # dictionary to compare model accuracy
    dict_models = {}
    best_one = {'Best model file': 0}

    # For all models' pickle files, load 'em
    # and store the accuracy to be compared
    for files in os.listdir("."):
        if files.endswith(".pkl"):
            print('Loaded %s:' % files, end=" ")
            print('Accuracy:', end=" ")
            dict_models[files] = pickle.load(open(files, 'rb'))['TestAcc']
            print(dict_models[files], "%")

    # Sort w.r.t testing accuracy of all models
    best_one['Best model file'] = sorted(dict_models.items(), key=op.itemgetter(1),
                                         reverse=True)[0][0]

    # Save if better model found
    if not best_one['Best model file'] == "best_model.pkl":
        print("\nBest accuracy is achieved by", best_one)
        best_model_file = best_one['Best model file']
        print("Saving best model ...", end=" ")
        if not replace:
            call("cp " + best_model_file + " " + best_model_file + ".orig", shell=True)
            call("mv " + best_model_file + ".orig" + " best_model.pkl", shell=True)
        else:
            call("mv " + best_model_file + " best_model.pkl", shell=True)
        print("done.")
    else:
        print("No better models. Train better dude.")

    # Get back to root directory
    os.chdir("../../")
    return
