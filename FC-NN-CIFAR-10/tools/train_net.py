""" Training code for new/saved models """

# System imports
from __future__ import print_function

import torch
from termcolor import colored

# Custom imports
import libs.nn as nnc
from libs.check_args import arguments, using_gpu

from data import dataset as dset

from vision.transforms import TransformData, see

from model_store import save_model
import create


# Training
def train(model=None):

    args = arguments()

    if model is None:
        model = create.create_model()
    
    print("\n+++++     TRAINING     +++++\n")

    model.show_log(arch=True, train=True)

    # Get data
    train_dataset = dset.CIFAR10(directory='data', download=True, 
                    train=True)

    # Data augmentation
    # Horizhontal flips. Giving the best results
    train_dataset = TransformData(train_dataset, transform='flipLR')
    # Upside-down flips. Average results
    # train_dataset = TransformData(train_dataset, transform='flipUD')
    # Crops. Bad results
    # train_dataset = TransformData(train_dataset, transform='crop')
    # Rotate image 90*times
    train_dataset = TransformData(train_dataset, transform='rotate90')
    print("Training set size:", len(train_dataset.data), "images.")

    # Optimizer
    optimizer = nnc.Optimize(model)
    print("\n# Stochastic gradient descent #")
    print("Learning rate: %.4f\n" % model.lr)
    for epoch in range(model.epochs):
        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        # Prepare batches from whole dataset
        train_loader = dset.data_loader(train_dataset.data, 
                       batch_size=dset.CIFAR10.batch_size, shuffled=True)
        # Iterate over batches
        for images, labels in train_loader:
            if using_gpu():
                images = images.cuda()
            # Training round
            model.train(images, labels)
            # Clear cache if using GPU (Unsure of effectiveness)
            if using_gpu():
                torch.cuda.empty_cache()
        print(colored('# Training Loss:', 'red'), end=" ")
        print('[%.4f] @ L.R: %.4f' % (model.loss, model.lr))
        model.loss_history.append(model.loss)
        optimizer.time_decay(epoch, 0.005)
        optimizer.set_optim_param(epoch)
    
    model.plot_loss('Training loss')
    
    # Model status
    model.model_trained = model.optimum['Trained'] = True        
    print("\nModel status:")
    print("{ Fitting tested:", model.optimum['Fitting tested'], "|", 
          "Trained:", model.optimum['Trained'], "|", 
          "Tested:", model.optimum['Tested'], "|", "Inferenced:", 
          model.optimum['Inferenced'], "}")
    print("{ Loss:", model.optimum['Loss'], "}\n")
    
    model.set_logs()        
    # Saving fitted model    
    if args.SAVE:
        save_model(args.SAVE, model)
    else:
        f = raw_input("Do you want to save the model? (y)es/(n)o: ").lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            save_model('model.pkl', model)
        else:
            print('Not saving model.')

    return model
