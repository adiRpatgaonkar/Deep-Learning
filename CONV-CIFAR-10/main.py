""" 
Using pickle data set CIFAR-10

Friday 16 February 2018 05:02:12 PM IST 

Created on Fri Feb 16 00:11:58 2018

@author: apatgao
"""
# imports for system library
from __future__ import print_function
import os, torch, numpy as np
from termcolor import colored
from subprocess import call
# Custom imports
import Dataset as dset, nnCustom as nnc, argsdo as do

"""
TODO
1. Optimize memory usage
"""
if (not do.model_load) and (not do.model_fit) and (not do.model_train) and (not do.model_test):
    print('Did nothing.')
    exit()
call('clear', shell=True)
if do.using_gpu and torch.cuda.is_available():
    torch.cuda.set_device(0)  # Subject to change
    print('\nUsing GPU: %d' % torch.cuda.current_device())
else:
    print('\nUsing CPU.')

global train_dataset, test_dataset, images, labels, train_loader, test_loader, optimizer # Global variables
filename = 'optimum.pkl'


def main():
    """ Creates, uses, trains and tests a NN model """
    
    # Define the network
    print('\n' + '+' * 16, '\nDefining network\n' + '+' * 16)
    model = nnc.ModelNN()
    model.add(nnc.Linear(32 * 32 * 3, 1024))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.Linear(1024, 512))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.Linear(512, 10))
    model.add(nnc.Criterion('Softmax'))
    model.show_net()

    if do.model_fit:
        # Model fitting test
        print("\n+++++Model fitting+++++\n")
        train_dataset = dset.CIFAR10(directory='data/', download=True, train=True)  # Get data
        # Hyper parameters
        model.epochs, model.lr = 10000, 0.2
        optimizer = nnc.Optimize(model)
        print("Learning rate: %.4f\n" % model.lr)
        fitting_loader = dset.data_loader(train_dataset.data, batch_size=dset.CIFAR10.batch_size, model_testing=True)
        for epoch in range(model.epochs):
           print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
           for images, labels in fitting_loader:
               if do.using_gpu:
                   images = images.cuda()
               model.train(images, labels)
               if do.using_gpu:
                   torch.cuda.empty_cache()
           print(colored('# Fitting test Loss:', 'red'), end="")
           print('[%.4f] @ L.R: %.9f' % (model.loss, model.lr))
           model.loss_history.append(model.loss)
           optimizer.time_decay(epoch, 0.0005)
           optimizer.set_optim_param(epoch)
        model.plot_loss()   
        for images, labels in fitting_loader:
            if do.using_gpu:
                images = images.cuda() 
            model.test(images, labels)
        labels = torch.from_numpy(np.array(labels))
        model.train_acc = model.optimum['TrainAcc'] = \
        (torch.mean((model.predictions == labels).float()) * 100)  # Training accuracy
        print("\nTraining accuracy = %.2f %%" % model.train_acc)
        model.loss_history = []

    if do.model_train:
        # Training
        print("\n+++++Training+++++\n")
        if not do.model_fit:
            train_dataset = dset.CIFAR10(directory='data/', download=True, train=True)  # Get data
        # Hyper parameters
        model.epochs, model.lr = 50, 0.08
        optimizer = nnc.Optimize(model)    
        print("\n# Stochastic gradient descent #")
        print("Learning rate: %.4f\n" % model.lr)
        for epoch in range(model.epochs):
            print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
            train_loader = dset.data_loader(train_dataset.data, batch_size=dset.CIFAR10.batch_size, shuffled=True)
            for images, labels in train_loader:
                if do.using_gpu:
                    images = images.cuda()
                model.train(images, labels)
                if do.using_gpu:
                    torch.cuda.empty_cache()
            #  print('# Training loss:', end=" ")
            print(colored('# Training Loss:', 'red'), end=" ")
            print('[%.4f] @ L.R: %.4f' % (model.loss, model.lr))
            model.loss_history.append(model.loss)
            optimizer.time_decay(epoch, 0.005)
            optimizer.set_optim_param(epoch)
        nnc.save_model(filename, model)
        
    if do.model_test:
        # Testing
        print("\n+++++++Testing+++++++\n")
        test_dataset = dset.CIFAR10(directory='data/', download=True, test=True)  # Get data
        if not do.model_train:
            optimizer = nnc.Optimize(model)
            if os.path.isfile(filename): # Load a saved model?
                t = raw_input('Test a previously model? (y)es/(n)o: model? (y)es/(n)o: ').lower()
                if t.lower() == 'y' or t.lower() == 'yes':
                    nnc.load_model(filename, model)
        test_loader = dset.data_loader(test_dataset.data, batch_size=dset.test_size, shuffled=False)
        for images, labels in test_loader:
            if do.using_gpu:
                images = images.cuda()
            model.test(images, labels)
        labels = torch.from_numpy(np.array(labels))
        print(colored('\n# Testing Loss:', 'red'), end="")
        print('[%.4f]' % (model.loss))        
        model.test_acc = model.optimum['TestAcc'] = \
        (torch.mean((model.predictions == labels).float()) * 100)  # Testing accuracy
        print(colored('\nTesting accuracy:', 'green'), end="")
        print(" = %.2f %%" % model.test_acc)
        if do.model_train:
            if os.path.isfile(filename):
                t = nnc.load_model(filename)
                print('Loss:', t['Loss'], '|', 'Testing accuracy:', t['TestAcc'], '%')
                if t['TestAcc'] < model.test_acc:
                    print('\nThis is the best model.', end=" ")
                else:
                    print('\nBetter models exist.')
                nnc.save_model(filename, model)
        if len(model.loss_history) > 1:
            model.plot_loss(to_show=True)
        
        print('\n' + '-' * 15 + '\nViewing results\n' + '-' * 15)
        model.display_results(labels, images)
        
        print('\n' + '-' * 7 + '\nExiting\n' + '-' * 7)
        
    if do.using_gpu:
        torch.cuda.empty_cache()
       

if __name__ == '__main__':
    main()
