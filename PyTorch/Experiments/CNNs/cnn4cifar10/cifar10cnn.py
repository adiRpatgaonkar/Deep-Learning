# CIFAR-10 using CNN using Pytorch

from __future__ import print_function

import argparse

# Neural net libs
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from whatodo import args as do    # Argument parser
from models4cifar10 import *      # Models defined for cifar10
from evalcifar10 import evaluate  # Evaluation tasks for cifar10 

# Device setup
device = torch.device("cuda:" + do.gpu_id if do.gpu_id is not None and torch.cuda.is_available() else "cpu")
print("\nUsing", device, "\n")

ID = "cnn1"

if do.load:
    # model can be a state dict or nn.Module object
    model = torch.load(do.load)

    if not isinstance(model, nn.Module):
        print("Loading model from state dict ...", end=" ")
        # Create a new model & revive it's state
        cnn = Model(ID)  
        cnn.load_state_dict(model) 
        print("done.")
    else:
        print("Loaded saved model.")
        cnn = model
        del model
      
if do.train:
    # Hyper Params
    max_epochs = 5
    learning_rate = 0.0005

if do.train or do.test:
    # Batch size
    batch_size = 100
    # Data normalized with:
    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)

# Cifar10 dataset
# Training: train + test data
# Testing: test data only
if do.train: 
    # Train data augmentation
    print("\nDefined transforms for train data augmentation.\n")
    augmented_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std)])
    # Train dataset
    trainset = dsets.CIFAR10(root='./data/', 
        train=True, transform=augmented_train,
        download=True)
    # Train data loader (Input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
        batch_size=batch_size, shuffle=True)
if do.train or do.test:
    # Test data transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std)])
    # Test dataset
    testset = dsets.CIFAR10(root='./data/',
        train=False, transform=transform_test)
    # Test data loader (Input pipeline)
    test_loader = torch.utils.data.DataLoader(dataset=testset, 
        batch_size=batch_size, shuffle=False)


if do.train:
    if not do.load:
        # If not training a loaded model
        # Create a brand new model
        cnn = Model(ID)

    cnn.to(device)
    print("\nModel id -> {}".format(ID))
    print("{}\n".format(cnn))

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    num_batches = len(trainset)//batch_size
    best_acc = 0  # Best accuracy tracking
    print("Training starts ...\n") 
    for epoch in range(max_epochs):
        print("-"*58, "Epoch:[{}/{}]".format(epoch+1, max_epochs))
        for i, (images, labels) in enumerate(train_loader):
            if (i+1) < num_batches:
                # Train
                images = images.to(device)
                labels = labels.to(device)
                images = Variable(images)
                labels = Variable(labels) 
                outputs = cnn(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0: 
                    print ("Iter:[{}/{}] Error:{:.4f}".format(i+1, num_batches, loss.item()))
            elif (i+1) == num_batches:
                # Cross-validate
                total, cval_acc = evaluate(cnn, [(images, labels)], device, task="cross_val")
                print("Cross-val on {} images: {} %".format(total, cval_acc))
        # Validate (test set) & check for the best model
        total, pres_acc = evaluate(cnn, test_loader, device=device, task="test")
        print("Validation on {} images: {} %".format(total, pres_acc), end="; ")
        if best_acc < pres_acc:
            best_acc = pres_acc
            if do.bm:
                torch.save(cnn.state_dict(), "cifar10_cnn_best.pkl")
        print("Best val accuracy: {} %".format(best_acc)) 
        # Epoch end

if do.test:
    print("Testing starts ...")
    # Final testing of the model. Sanity check.
    total, accuracy = evaluate(cnn, test_loader, device, task="test")
    print("Accuracy of the {} on {} test images: {} %".format(ID, total, accuracy))

if do.fm:
    # Save final model
    torch.save(cnn.state_dict(), "cifar10_cnn_final.pkl")

