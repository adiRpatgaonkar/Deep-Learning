from __future__ import print_function

from copy import deepcopy  # Save snapshot of best model weights

# Neural net libs
import torch
import torch.nn as nn
from torch.autograd import Variable

from imagenetdata import get_dataset, get_loader, rgb_mean, rgb_std
from inferimagenet import see
from models4imagenet import models

# Device setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("\nUsing", device, "\n")

# Hyper Params
max_epochs = 2
learning_rate = 0.0005

# Batch size
batch_size = 100

# Imagenet dataset
# Train dataset
trainset = get_dataset("train")
valset = get_dataset("test")

# Data loaders
train_loader = get_loader("train", trainset)
val_loader = get_loader("test", valset, s=False)

# Build model
ID = "cnn4"
cnn = models(ID)
cnn.to(device)
print("\nModel id -> {}".format(ID))
print("{}\n".format(cnn))

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

for i, (images, labels) in enumerate(train_loader):
    if i == 0:
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        break

'''
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
    see(cnn.conv1[0].weight[15].detach(), mean=rgb_mean, std=rgb_std, title="Conv layer 1 last kernel (initial)")
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
    print("Accuracy of {} on {} test images: {} %".format(ID, total, accuracy))

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
see(cnn.conv1[0].weight[15].detach(), mean=rgb_mean, std=rgb_std, title="Conv layer 1 last kernel (trained)")
'''
