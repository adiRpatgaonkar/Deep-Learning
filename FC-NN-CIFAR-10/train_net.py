from __future__ import print_function
from termcolor import colored
import torch
import yaml

import do_stuff as do
import nnCustom as nnc
import Dataset as dset


def set_hyper_paramters(config):

    with open(config, 'r') as f:
        cfg = yaml.load(f)

    model.type += cfg["MODEL"]["TYPE"]

    model.weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
    model.reg = cfg["SOLVER"]["REG"]

    model.lr = cfg["TRAIN"]["BASE_LR"]
    model.lr_policy += cfg["TRAIN"]["LR_POLICY"]
    model.decay_rate = cfg["TRAIN"]["DECAY_RATE"]
    model.epochs = cfg["TRAIN"]["EPOCHS"]


def create_model():
    # Define the network
    print('\n' + '+' * 16, '\nDefining network\n' + '+' * 16)
    global model
    model = nnc.ModelNN()
    set_hyper_paramters(do.args.CFG)
    model.add(nnc.LinearLayer(32 * 32 * 3, 1024))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(1024, 512))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(512, 10))
    model.add(nnc.CeCriterion('Softmax'))
    return model

def train():

    model = create_model()

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