""" 
Fetch, shuffle, normalize data set.

@author: apatgao
"""
# imports for system library
# import sys
from __future__ import print_function
import pickle, os
from subprocess import call
from random import shuffle
import torch

global images, labels, data

def data_loader(data_set, batch_size, model_testing= False, shuffled=False):
    """ Shuffles given data x"""
    if shuffled:
        shuffle(data_set)
    images, labels = [], []
    for x, y in data_set:
        images.append(x)
        labels.append(y)
    images = torch.stack(images, dim=0)
    if model_testing:
      num_batches = 1
    else:  
      num_batches = len(data_set) / batch_size
    mini_batches = []
    for batch in range(num_batches):
        b_start = batch * batch_size
        b_end = (batch + 1) * batch_size
        mini_batches.append((images[b_start:b_end], labels[b_start:b_end]))
        
    return mini_batches

class CIFAR10():
    ### +++ Data info found here +++ ###
    data_size = 60000
    train_size = 50000
    batch_size = 100
    train_batches = train_size / batch_size
    test_size = 10000
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(self, directory='.', download=False, train=False, test=False):
        ## +++ Data info found here +++ ###
        self.data = []
        self.images = []
        self.labels = []
        self.dir = directory
        if download:
            self.download_cifar10()
        self.get_dataset(train, test)
        
            
    def download_cifar10(self):
        """ If dataset does not exist, downloads it """
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        if not os.path.exists(self.dir):
            print("Creating data directory.")
            call(["mkdir", self.dir])
        os.chdir(os.getcwd() + '/' + self.dir)
        # Download dataset
        if not os.path.isfile("cifar-10-python.tar.gz"):
            print('\n' + '-' * 20, '\nDownloading data ...\n' + '-' * 20)
            call(["wget", url])
        else:
            print('Dataset already downloaded.')
        # Extract dataset
        if (not os.path.isfile("data_batch_1"))\
        or (not os.path.isfile("data_batch_2"))\
        or (not os.path.isfile("data_batch_3"))\
        or (not os.path.isfile("data_batch_4"))\
        or (not os.path.isfile("data_batch_5"))\
        or (not os.path.isfile("test_batch")):
            print('\n' + '-' * 22, '\nExtracting dataset ...\n' + '-' * 22)
            call("tar -xzf cifar-10-python.tar.gz", shell=True)
            call("mv cifar-10-batches-py/* .", shell=True)
            call("rm -r cifar-10-batches-py", shell=True)
        else:
            print('\nDataset already downloaded and set up.\n')
        os.chdir('..')
        
    def get_dataset(self, train=False, test=False):
        """ Obtains training or testing data from pickle files """
    
        print('Fetching', end=" ")
        if train:
            print('training data ...')
            og_num_batches = 5
        else:  # test
            print('testing data ...')
            og_num_batches = 1
        og_batch_size = 10000
        self.labels = []
        for batch in range(og_num_batches):
            if train:
                data_file = open('./' + self.dir + 'data_batch_' + str(batch + 1), 'rb')
            elif test:
                data_file = open('./' + self.dir + 'test_batch', 'rb')
        
            tuples = pickle.load(data_file)
            data_file.close()
        
            image_data = tuples['data'].reshape(og_batch_size, 3, 32, 32).astype("uint8")
            
            if batch == 0:
                self.images = torch.from_numpy(image_data).type(torch.FloatTensor)
            else:
                self.images = torch.cat((self.images, torch.from_numpy(image_data).type(torch.FloatTensor)), 0)
            for label in tuples['labels']:
                self.labels.append(label)
        
        for i, l in zip(self.images, self.labels):
            self.data.append((i, l))
    
        return self.data
