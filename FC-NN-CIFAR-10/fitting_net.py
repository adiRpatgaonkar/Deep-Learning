from __future__ import print_function
from termcolor import colored
import yaml
import torch
import numpy as np

import do_stuff as do
import nnCustom as nnc
import Dataset as dset
import create

def fit():

    model = create.create_model()

    # Model fitting test
    print("\n+++++Model fitting+++++\n")
    # Get data
    train_dataset = dset.CIFAR10(directory='data/', download=True, train=True)  
    optimizer = nnc.Optimize(model)
    print("Learning rate: %.4f\n" % model.lr)
    fitting_loader = dset.data_loader(train_dataset.data, batch_size=dset.CIFAR10.batch_size, model_testing=True)
    for epoch in range(model.epochs):
        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        for images, labels in fitting_loader:
            if do.args.GPU:
                images = images.cuda()
            model.train(images, labels)
            if do.args.GPU:
                torch.cuda.empty_cache()
        print(colored('# Fitting test Loss:', 'red'), end="")
        print('[%.4f] @ L.R: %.9f' % (model.loss, model.lr))
        model.loss_history.append(model.loss)
        optimizer.time_decay(epoch, 0.0005)
        optimizer.set_optim_param(epoch)
    model.plot_loss()
    if do.args.SAVE:
        nnc.save_model('model_non_tested.pkl', model)
    return model, fitting_loader
    '''
    for images, labels in fitting_loader:
        if do.args.GPU:
            images = images.cuda()
        model.test(images, labels)
    labels = torch.from_numpy(np.array(labels))
    model.train_acc = model.optimum['TrainAcc'] = \
        (torch.mean((model.predictions == labels).float()) * 100)  # Training accuracy
    print("\nTraining accuracy = %.2f %%" % model.train_acc)
    model.loss_history = []
    '''