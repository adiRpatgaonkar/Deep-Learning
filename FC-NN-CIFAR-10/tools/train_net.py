""" Training code for new/saved models """

# System imports
from __future__ import print_function

import numpy as np
import torch
from termcolor import colored

# Custom imports
from church import nn
from libs.check_args import arguments, using_gpu
from data import dataset as dset
from model_store import save_model
from tools import create
from vision.transforms import Transforms


# Training
def train_model(model=None):
    global images, ground_truths

    args = arguments()

    if model is None:
        model = create.create_model()

    print("\n+++++     TRAINING     +++++\n")
    model.show_log(arch=True, train=True)

    # Get train data for training and cross validation
    train_dataset = dset.CIFAR10(directory='data',
                                 download=True,
                                 train=True)
    # Data augmentation
    train_dataset = Transforms(dataset=train_dataset,
                               lr_flip=True)

    # Get validation data
    val_dataset = dset.CIFAR10(directory='data',
                               download=True,
                               test=True)
    # Validation data in batches
    val_loader = dset.data_loader(data=val_dataset.data,
                                  batch_size=dset.CIFAR10.test_size,
                                  shuffled=False)

    # Optimizer/Scheduler
    optimizer = nn.Optimize(model)

    # +++++ Epoch start +++++ #
    for model.curr_epoch in range(model.max_epochs):
        # Prepare batches from whole dataset
        train_loader = dset.data_loader(data=train_dataset.data,
                                        batch_size=dset.CIFAR10.batch_size,
                                        shuffled=True)
        # Iterate over batches
        for i, (images, ground_truths) in enumerate(train_loader[:-1]):
            if using_gpu():
                images = images.cuda()
            # Training round
            model.train(images, ground_truths)
            # Print iterations over batches
            # print("Iter: [%d/%d]" % (i, len(train_loader)), end=" ")
            # print("# Training Loss: [%.4f] @ L.R: %.5f" % (model.train_loss, model.lr), end=" ")

            # Clear cache if using GPU (Unsure of effectiveness)
            if using_gpu():
                torch.cuda.empty_cache()

        print('Epoch: [%d/%d]' % (model.curr_epoch + 1, model.max_epochs), end=" ")
        print("@ [L.R: %.4f]" % model.lr)
        # Print training loss after every epoch
        print(colored('# Training loss:', 'red'), end=" ")
        print('[%.4f]' % model.train_loss)
        model.train_loss_history.append(model.train_loss)
        
        # +++++ Cross validation over a portion of train set +++++ #
        for images, ground_truths in train_loader[-2:-1]:
            if using_gpu():
                images = images.cuda()
            model.test(images, ground_truths)
        ground_truths = torch.from_numpy(np.array(ground_truths))
        # Training accuracy after every epoch
        model.train_acc = torch.mean((model.predictions == ground_truths).float()) * 100
        print(colored('# Cross validation accuracy:', 'red'), end=" ")
        print("[%.2f%%]" % model.train_acc)
        model.crossval_acc_history.append(model.train_acc)

        # +++++ Validation over test set +++++ #
        # For every epoch check validation accuracy
        for images, ground_truths in val_loader:
            if using_gpu():
                images = images.cuda()
            model.test(images, ground_truths)
            # Clear cache if using GPU (Unsure of effectiveness)
            if using_gpu():
                torch.cuda.empty_cache()
        # Print validation loss over test set
        print(colored('# Validation loss:', 'green'), end=" ")
        print('[%.4f]' % model.test_loss)
        model.test_loss_history.append(model.test_loss)
        ground_truths = torch.from_numpy(np.array(ground_truths))
        # Validation accuracy
        model.test_acc = torch.mean((model.predictions == ground_truths).float()) * 100
        print(colored('# Validation accuracy:', 'green'), end="")
        print("[%.2f%%]" % model.test_acc, end=" ")
        print("# Best val accuracy: [%.2f%%]" % model.optimum['Testing-accuracy'])
        model.test_acc_history.append(model.test_acc)

        # +++++ L.R. schedule & store best params +++++ #
        optimizer.set_optim_param()
        optimizer.time_decay()

        # +++++ Class performance @ each epoch +++++ #
        class_performance = [0.0] * dset.CIFAR10.num_classes
        for predicted, gt in zip(model.predictions, ground_truths):
            if predicted == gt:
                class_performance[gt] += 1
        for i, c in enumerate(dset.CIFAR10.classes):
            print("%s:%.1f%% |" % (c, 100 * (class_performance[i] / dset.CIFAR10.imgs_per_class)), end=" ")
        print("\n")
    # +++++ Epoch end +++++ #

    # Model status
    model.trained = True
    # Plot training & validation, show & set logs.
    # model.plot_history(loss_history=True, accuracy_history=True)
    model.save_state()
    model.show_log(curr_status=True)
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
