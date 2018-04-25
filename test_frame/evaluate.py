from __future__ import print_function

import torch

import cutorch
import cutorchvision.datasets as dsets


accuracy = {'cval': [], 'test': []}

def evaluate(model, dataset, task):
    global images, labels, outputs
    global total
    global correct
    global accuracy
    model.eval()  # Switch: evaluation mode
    correct = 0
    total = 0
    if isinstance(dataset, tuple) and task == "cross-validate":
        dataset = [dataset]  # Usually for cross-validation
    for images, labels in dataset:
        if cutorch.gpu.used:
            images = images.cuda()  # Move image batch to GPU
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = outputs.data
        total += len(labels)
        correct += (predicted == labels).sum()
        if cutorch.gpu.used:  # GPU cache cleaning. Unsure of effectiveness
            torch.cuda.empty_cache()
    if task == "cross-validate":
        # ++++ Cross validation accuracy ++++ #
        accuracy['cval'].append(100 * correct / total)  # To plot
        return total, accuracy['cval'][-1]
    elif task == "test":
        # ++++ Test Accuracy ++++ #
        accuracy['test'].append(100 * correct / total)  # To plot
        model.results['accuracy'] = accuracy['test'][-1]
        # ++++ Class performance ++++ #
        print("\nClass performance:")
        class_performance = [0] * dsets.CIFAR10.num_classes
        for pred, gt in zip(predicted, labels):
            if pred == gt:
                class_performance[gt] += 1
        for i, c in enumerate(dsets.CIFAR10.classes):
            print("{}: {}% |".format(c, 100*class_performance[i]/dsets.CIFAR10.images_per_class), end=" ")
        return total
