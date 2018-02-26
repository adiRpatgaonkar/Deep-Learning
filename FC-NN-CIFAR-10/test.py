from __future__ import print_function
import os
from subprocess import call


url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
dir = 'data/'

if not os.path.exists(dir):
    call(["mkdir", dir])
os.chdir(os.getcwd() + '/' + dir)
if not os.path.isfile("cifar-10-python.tar.gz"):
    print('\n', '-' * 20, '\n Downloading data ...\n', '-' * 20)
    call(["wget", url])
else:
    print('\n', '-' * 22, '\n Extracting dataset ...\n', '-' * 22)
    call("tar -xzf cifar-10-python.tar.gz", shell=True)
    call("mv cifar-10-batches-py/* .", shell=True)
    call("rm -r cifar-10-batches-py", shell=True)
print('\n', '-' * 5, '\n Done.\n', '-' * 5)
