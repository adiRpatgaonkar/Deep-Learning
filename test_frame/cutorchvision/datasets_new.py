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
import numpy as np

__dlevel__ = 0


class CIFAR10:
    # +++ Data info found here +++ #
    name = 'cifar10'
    data_size = 60000
    train_size = 50000
    test_size = 10000
    images_per_class = 1000  # test set
    classes = ['airplane', 'automobile',
               'bird', 'cat',
               'deer', 'dog',
               'frog', 'horse',
               'ship', 'truck']
    num_classes = len(classes)

    def __init__(self, directory='.', download=False, train=False, test=False, form=None):
        """ Setup necessary variables for Cifar10 dataset """
        if form == "ndarray" or form == "tensor":
            self.form = form
            self.data = []
            if form == "tensor":
                self.images = torch.FloatTensor()
                self.labels = torch.LongTensor()
            elif form == "numpy":
                self.images = np.array([])
                self.labels = np.array([])

            self.dir = directory
            self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            if download:
                self.download_cifar10()
            self.get_dataset(train, test)
        else:
            self.data = None
            print("Specify appropriate data format.")
            print("Not fetching data.")

    @staticmethod
    def verify_setup():
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
            call("mkdir " + self.dir, shell=True)
        if not os.path.exists(self.dir + '/cifar10'):
            call("mkdir " + self.dir + "/cifar10", shell=True)
        os.chdir(os.getcwd() + '/' + self.dir + '/cifar10/')

        if __debug__:
            if __dlevel__ == 1:
                print(os.getcwd())
                
        # Download dataset
        if not self.verify_setup() and not os.path.isfile("cifar-10-python.tar.gz"):
            print('\n' + '-' * 20, '\nDownloading data ...\n' + '-' * 20)
            call("wget " + self.url, shell=True)
        else:
            print('\nDataset already downloaded.')
        # Extract dataset
        if not self.verify_setup():
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
            batch_files = 5
        else:
            print('testing data', end=" ")
            batch_files = 1
        og_batch_size = 10000

        if __debug__:
            if __dlevel__ == 1:
                print('CWD: {}'.format(os.getcwd()))

        print("from " + self.dir + " ...", end=" ")

        for batch in range(batch_files):
            if train:
                data_file = open('./' + self.dir + '/cifar10/' + 'data_batch_' + str(batch + 1), 'rb')
            elif test:
                data_file = open('./' + self.dir + '/cifar10/' + 'test_batch', 'rb')

            tuples = pickle.load(data_file)
            data_file.close()

            # Originally in numpy ndarray form
            image_data = tuples['data'].reshape(og_batch_size, 3, 32, 32).astype("uint8")
            label_data = tuples['labels']
            
            if self.form == "tensor":
                self.images = torch.cat((self.images, torch.from_numpy(image_data).float()), 0)
                self.labels = torch.cat((self.labels, torch.LongTensor(label_data)), 0)
            elif self.form == "numpy":
                self.images = np.vstack([self.images, image_data], axis=0)
                self.labels = np.vstack([self.labels, label_data], axis=0)

            # for label in label_data:
            #     self.labels.append(label)

        # print(self.images.size(), self.labels.size())
        for image, label in zip(self.images, self.labels):
            if self.form == "tensor":
                label = torch.LongTensor([label])
            self.data.append((image, label))
        # print(self.data[0])

        print("done.\n")
        return self.data