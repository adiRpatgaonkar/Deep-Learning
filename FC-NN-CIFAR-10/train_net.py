from __future__ import print_function
from termcolor import colored
import torch
import yaml

import do_stuff as do
import nnCustom as nnc
import Dataset as dset
import create


def train():

    model = create.create_model()

    # Training
    print("\n+++++Training+++++\n")
    train_dataset = dset.CIFAR10(directory='data/', download=True, train=True)  # Get data
    # Optimizer
    optimizer = nnc.Optimize(model)
    print("\n# Stochastic gradient descent #")
    print("Learning rate: %.4f\n" % model.lr)
    for epoch in range(model.epochs):
        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        train_loader = dset.data_loader(train_dataset.data, batch_size=dset.CIFAR10.batch_size, shuffled=True)
        for images, labels in train_loader:
            if do.args.GPU:
                images = images.cuda()
            model.train(images, labels)
            if do.args.GPU:
                torch.cuda.empty_cache()
        print(colored('# Training Loss:', 'red'), end=" ")
        print('[%.4f] @ L.R: %.4f' % (model.loss, model.lr))
        model.loss_history.append(model.loss)
        optimizer.time_decay(epoch, 0.005)
        optimizer.set_optim_param(epoch)

    if do.args.SAVE:
        nnc.save_model('model_non_tested.pkl', model)

    return model