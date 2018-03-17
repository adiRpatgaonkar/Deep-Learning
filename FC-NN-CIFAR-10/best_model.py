""" Search, compare and find the best model """

from __future__ import print_function
import os
import pickle
from subprocess import call

# Change directory to ./outputs/models
os.chdir("./outputs/models")

# dictionary to compare model accuracy
dict_models = {}
# For all models' pickle files, load 'em
# and store the accuracy to be compared
for files in os.listdir("."):
    if files.endswith(".pkl"):
        print('Loading %s ...' % files, end=" ")
        dict_models[files] = pickle.load(open(files, 'rb'))['TestAcc']
        print("done.")
        
best_one = {'Best model file': ""}
for key in sorted(dict_models.keys()):
    print("Model file: [%s] Accuracy: [%.2f]" % (key, dict_models[key]))
    if best_one['Acc'] < dict_models[key]:
        print('Better model found')
        best_one['Acc'] = key
        
print("Best accuracy is achieved by", best_one)
print(best_one)
best_model_file = best_one['Acc']
print("Saving best model ... ")
call("cp " + best_model_file + " " + best_model_file + ".orig", shell=True)
call("mv " + best_model_file + ".orig" + " best_model.pkl", shell=True)
# Return to root dictionary
os.chdir("../..")
