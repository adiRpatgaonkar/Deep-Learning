from __future__ import print_function

import torch


classes = ('airplane', 'automobile',
           'bird', 'cat',
           'deer', 'dog',
           'frog', 'horse',
           'ship', 'truck')
ims_per_class = 1000

def evaluate(model, dataset, device, task="test", vis=None):
    assert task == "test" or task == "cross_val", \
           "Invalid task for evaluation. Expected test/cross_val"
    if device != "cpu":
        if not next(model.parameters()).is_cuda:
            model.to(device)
    model.eval()  # Switch: evaluation mode
    correct = 0
    total = 0 
    if task == "test":
        class_performance = [0.0] * len(classes)
    with torch.no_grad():
        for images, labels in dataset:
            images = images.to(device)  # Move image batch to GPU
            labels = labels.to(device)
            outputs = model(images, vis=vis)
            # Softmax classifier
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum() 
            if task == "test":  
                # Gather class performance  
                for pred, gt in zip(predicted, labels):
                    if pred == gt:
                        class_performance[gt] += 1
            if device != "cpu":  # GPU cache cleaning. Unsure of effectiveness
                torch.cuda.empty_cache()
        if task == "test":
            # Class accuracy
            print("Class performance:\n" + "-"*20)
            for i, c in enumerate(classes): 
                print("{0:11}: {1:5} %".format(c, 100*class_performance[i]/ims_per_class))
        accuracy = 100 * correct / total
        #print("") 
    return total, accuracy
