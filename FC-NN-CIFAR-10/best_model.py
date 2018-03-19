""" Search, compare and find the best model """

# System imports
from __future__ import print_function
import os
import pickle
from subprocess import call


os.chdir("outputs/models/")

# dictionary to compare model accuracy
dict_models = {}
best_one = {'Best model file': 0}

# For all models' pickle files, load 'em
# and store the accuracy to be compared
for files in os.listdir("."):
    if files.endswith(".pkl"):
        print('Loading %s.' % files)
        dict_models[files] = pickle.load(open(files, 'rb'))['TestAcc']

# Compare testing accuracy of all models
for key in sorted(dict_models.keys()):
    print("Model file: [%s] Accuracy: [%.2f]" % (key, dict_models[key]))
    print(best_one['Best model file'], dict_models[key])
    if best_one['Best model file'] < dict_models[key]:
        # print('Better model found', key)
        best_one['Best model file'] = key

# Save if better model found
if not best_one['Best model file'] == 0:
    print("\nBest accuracy is achieved by", best_one)
    best_model_file = best_one['Best model file']
    print("Saving best model ... ")
    call("cp " + best_model_file + " " + best_model_file + ".orig", shell=True)
    call("mv " + best_model_file + ".orig" + " best_model.pkl", shell=True)
else:
    print("No better models. Train better dude.")

os.chdir("../../")