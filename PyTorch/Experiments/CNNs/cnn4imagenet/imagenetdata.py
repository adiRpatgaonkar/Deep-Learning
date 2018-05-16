from __future__ import print_function

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

def get_dataset(task):
    assert task in ("train", "test"), "Invalid task. Expected (train/test)"
    # Data normalized with:
    
    std_normalize = transforms.Normalize(rgb_mean, rgb_std)
    # Train data augmentation
    print("Defined transforms for " + task + " data augmentation.")
    augmented_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        std_normalize])
    augmented_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        std_normalize])

    print("Fetching " + task + "set ...", end=" ")
    if task == "train":
        dataset = dsets.ImageFolder('./data/train/', transform=augmented_train)
    elif task == "test":
        dataset = dsets.ImageFolder('./data/val/', transform=augmented_val)
    print("done.")
    return dataset

def get_loader(task, dataset, bs=128, s=True, pin_memory=True):
    assert task in ("train", "test"), "Invalid task. Expected (train/test)"
    print("Preparing dataloader for " + task + "set ...", end=" ")
    # Input pipeline
    loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_size=bs, shuffle=s, pin_memory=pin_memory,
        num_workers=4)
    print("done.")
    return loader