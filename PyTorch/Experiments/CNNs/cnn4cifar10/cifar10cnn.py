# CIFAR-10 using CNN using Pytorch

from __future__ import print_function

import argparse
from copy import deepcopy  # Save snapshot of best model weights

# Neural net libs
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from whatodo import args as do    # Argument parser
from models4cifar10 import models # Models defined for cifar10
from evalcifar10 import evaluate  # Evaluation tasks for cifar10 
from infercifar10 import see

# Device setup
device = torch.device("cuda:" + do.gpu_id if do.gpu_id is not None and torch.cuda.is_available() else "cpu")
print("\nUsing", device, "\n")

ID = do.mid

if do.load:
    # model can be a state dict or nn.Module object
    model = torch.load(do.load)

    if not isinstance(model, nn.Module):
        print("Loading model from state dict ...", end=" ")
        # Create a new model & revive it's state
        cnn = models(ID)  
        cnn.load_state_dict(model) 
        print("done.")
    else:
        print("Loaded saved model.")
        cnn = model
        del model

if do.train:
    # Hyper Params
    max_epochs = do.epochs
    learning_rate = do.lr

if do.train or do.test or do.infer:
    # Batch size
    batch_size = do.bs
    # Data normalized with:
    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)

# Cifar10 dataset
# Training & inference: train + test data
# Testing: test data only
if do.train or do.infer: 
    # Train data augmentation
    print("Defined transforms for train data augmentation.\n")
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
if do.train or do.test or do.infer:
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
        # If not training a "loaded" model
        # Create a brand new model
        cnn = models(ID)
    
    cnn.to(device)
    #see(cnn.conv1[0].weight[15].detach(), mean=rgb_mean, std=rgb_std, title="Conv layer 1 last kernel (initial)")
    print("\nModel id -> {}".format(ID))
    print("{}\n".format(cnn))

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

    net_iter = 0
    num_batches = len(trainset)//batch_size
    best_acc = 0  # Best accuracy tracking
    print("Training starts ...\n")
    print("Hypers:\nMax-epochs: {}\nLearning-rate: {}\nBatch-size: {}\n".format(max_epochs, learning_rate, batch_size))
    for epoch in range(max_epochs):
        print("-"*58, "Epoch:[{}/{}]".format(epoch+1, max_epochs))
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            net_iter += 1
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
                    print("Net_iter:[{}/{}]".format(net_iter, max_epochs*num_batches), end=" ") 
                    print("Iter:[{}/{}] Error:{:.4f}".format(i+1, num_batches, loss.item()))
            elif (i+1) == num_batches:
                # Cross-validate
                total, cval_acc = evaluate(cnn, [(images, labels)], device, task="cross_val")
                print("Cross-val on {} images: {} %".format(total, cval_acc))
        # Validate (test set)
        total, pres_acc = evaluate(cnn, test_loader, device=device, task="test")
        print("Validation on {} images: {} %".format(total, pres_acc), end="; ")
        # Check for the best model weights
        if best_acc < pres_acc:
            best_acc = pres_acc
            # Save the best model weights-snapshot
            if do.bm:
                best_model_wts = deepcopy(cnn.state_dict())
                best_iter = net_iter
        print("Best val accuracy: {} %".format(best_acc))
        # Save the best & final model weights
        if do.bm and (epoch+1) == max_epochs:
            torch.save(best_model_wts, "cifar10_"+ID+"_best_wts_"+str(best_iter)+".pkl")
        if do.fm and (epoch+1) == max_epochs:
            torch.save(cnn.state_dict(), "cifar10_"+ID+"_final_wts_"+str(net_iter)+".pkl")
        print("")
	    # Unsure of effectiveness
        if device != "cpu":
            torch.cuda.empty_cache()
        # Epoch end

if do.test:
    print("Testing starts ...")
    # Final testing of the model. Sanity check.
    total, accuracy = evaluate(cnn, test_loader, device, task="test")
    #print(cnn.__dict__.keys())
    #see(cnn.out_buffer[3][0][0], title="cnn weights 6th conv layer")
    print("Accuracy of {} on {} test images: {} %".format(ID, total, accuracy))


#see(cnn.conv1[5].weight[15].detach(), mean=rgb_mean, std=rgb_std, title="Conv layer 1 last kernel (trained)")
