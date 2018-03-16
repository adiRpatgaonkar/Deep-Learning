""" Search, compare and find the best model """

from __future__ import print_function
import os
import pickle
from subprocess import call

# Change directory to ./outputs/models
os.chdir("./outputs/models")

# dictionary to compare model accuracy
m_dict = {}
# For all models' pickle files, load 'em
# and store the accuracy to be compared
for files in os.listdir("."):
    if files.endswith(".pkl"):
        print('Loading %s ...' % files, end = " ")
        t = pickle.load(open(files, 'rb'))
        m_dict[files] = t['TestAcc']
        print("done.")
        
best_one = {'Acc': 0}
for key in sorted(m_dict.keys()):
    print("Model file: [%s] Accuracy: [%.2f]" % (key, m_dict[key]))
    if best_one['Acc'] < m_dict[key]:
        print('Better model found')
        best_one['Acc'] = key
        
print("Best accuracy is achieved by", best_one)
print(best_one)
best_model_file = best_one['Acc']
print("Saving best model ... ")
call("cp " + best_model_file + " " + best_model_file + ".orig" , shell=True)
call("mv " + best_model_file + ".orig" + " best_model.pkl", shell=True)

