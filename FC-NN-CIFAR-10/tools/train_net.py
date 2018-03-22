""" Training code for new/saved models """

# System imports
from __future__ import print_function
from termcolor import colored
import torch

# Custom imports
from libs.check_args import arguments, using_gpu
import libs.nn as nnc
from data import dataset as dset
from vision.transforms import Transforms, see
from tools import create
from model_store import save_model



# Training
def train(model=None):

    args = arguments()

    if model is None:
        model = create.create_model()
    
    print("\n+++++     TRAINING     +++++\n")

    model.show_log(arch=True, train=True)

    # Get data
    train_dataset = dset.CIFAR10(directory='data', 
        download=True, 
        train=True)

    # Data augmentation
    train_dataset = Transforms(
        dataset=train_dataset,
        lr_flip=True,
        rotate90=True, times=1)

    # Size after augmentation
    print("Training set size:", len(train_dataset.data), "images.")

    # Optimizer/Scheduler
    optimizer = nnc.Optimize(model)

    # SGD
    print("\n# Stochastic gradient descent #")
    print("Learning rate: %.4f\n" % model.lr)

    # Epochs
    for epoch in range(model.epochs):

        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        # Prepare batches from whole dataset
        train_loader = dset.data_loader(data=train_dataset.data, 
            batch_size=dset.CIFAR10.batch_size, 
            shuffled=True)
        # Iterate over batches
        for images, labels in train_loader:
            if using_gpu():
                images = images.cuda()
            # Training round
            model.train(images, labels)
            # Clear cache if using GPU (Unsure of effectiveness)
            if using_gpu():
                torch.cuda.empty_cache()
        # Print training loss
        print(colored('# Training Loss:', 'red'), end=" ")
        print('[%.4f] @ L.R: %.4f' % (model.loss, model.lr))
        model.loss_history.append(model.loss)

        optimizer.time_decay(epoch, model.decay_rate)
        optimizer.set_optim_param(epoch)
    
    # model.plot_loss('Training loss')
    
    # Model status
    model.trained = True
    
    model.show_log(curr_status=True)
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
