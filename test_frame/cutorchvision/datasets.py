"""
Fetch, setup, shuffle data set.

@author: apatgao
"""

# System imports
from __future__ import print_function

import os
import pickle
from subprocess import call

import torch

__dlevel__ = 0


class CIFAR10:
    # +++ Data info found here +++ #

    data_size = 60000
    train_size = 50000
    batch_size = 100
    test_size = 10000
    images_per_class = 1000  # test set
    classes = ['airplane', 'automobile',
               'bird', 'cat',
               'deer', 'dog',
               'frog', 'horse',
               'ship', 'truck']
    num_classes = len(classes)

    def __init__(self, directory='.', download=False, train=False, test=False):
        """ Setup necessary variables for Cifar10 dataset """
        self.name = 'cifar10'
        self.data = []
        self.images = []
        self.labels = []
        self.dir = directory
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        if download:
            self.download_cifar10()
        self.get_dataset(train, test)

    @property
    def verify_setup(self):
        if (os.path.isfile("data_batch_1")) \
                and (os.path.isfile("data_batch_2")) \
                and (os.path.isfile("data_batch_3")) \
                and (os.path.isfile("data_batch_4")) \
                and (os.path.isfile("data_batch_5")) \
                and (os.path.isfile("test_batch")):
            return True
        return False

    def download_cifar10(self):
        """ If dataset does not exist, download it """
        if __debug__:
            if __dlevel__ == 1:
                os.getcwd()

        if not os.path.exists(self.dir):
            print("Creating data directory.")
            call("mkdir " + self.dir)
        if not os.path.exists(self.dir + '/cifar10'):
            call("mkdir " + self.dir + "/cifar10", shell=True)
        os.chdir(os.getcwd() + '/' + self.dir + '/cifar10/')

        if __debug__:
            if __dlevel__ == 1:
                print(os.getcwd())

        # Download dataset
        if not os.path.isfile("cifar-10-python.tar.gz"):
            print('\n' + '-' * 20, '\nDownloading data ...\n' + '-' * 20)
            call("wget " + self.url, shell=True)
        else:
            print('\nDataset already downloaded.')
        # Extract dataset
        if not self.verify_setup:
            print('\n' + '-' * 22, '\nExtracting dataset ...\n' + '-' * 22)
            call("tar -xzf cifar-10-python.tar.gz", shell=True)
            call("mv cifar-10-batches-py/* .", shell=True)
            call("rm -r cifar-10-batches-py", shell=True)
        else:
            print('Dataset already set up.')
        os.chdir('../../../')

        if __debug__:
            if __dlevel__ == 1:
                print(os.getcwd())

    def get_dataset(self, train=False, test=False):
        """ Obtains training or testing data from pickle files """

        print('Fetching', end=" ")
        if train:
            print('training data', end=" ")
            og_num_batches = 5
        else:  # test
            print('testing data', end=" ")
            og_num_batches = 1
        og_batch_size = 10000
        self.labels = []

        if __debug__:
            if __dlevel__ == 1:
                print('CWD: {}'.format(os.getcwd()))

        print("from " + self.dir + " ...", end=" ")

        for batch in range(og_num_batches):
            if train:
                data_file = open('./' + self.dir + '/cifar10/' + 'data_batch_' + str(batch + 1), 'rb')
            elif test:
                data_file = open('./' + self.dir + '/cifar10/' + 'test_batch', 'rb')

            tuples = pickle.load(data_file)
            data_file.close()

            image_data = tuples['data'].reshape(og_batch_size, 3, 32, 32).astype("uint8")

            if batch == 0:
                self.images = torch.from_numpy(image_data).type(torch.FloatTensor)
            else:
                self.images = torch.cat((self.images, torch.from_numpy(image_data).type(torch.FloatTensor)), 0)
            for label in tuples['labels']:
                self.labels.append(label)

        for image, label in zip(self.images, self.labels):
            self.data.append((image, label))

        print("done.\n")

        return self.data
