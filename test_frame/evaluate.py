import torch
import cutorch

global total
global correct
global images, labels, outputs
global accuracy

accuracy = {'cval': [], 'test': []}

if cutorch.gpu_check.available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    using_gpu = True
else:
    using_gpu = False


def evaluate(model, dataset, task):
    model.eval()  # Switch: evaluation mode
    correct = 0
    total = 0
    for images, labels in dataset:
        if using_gpu:
            images = images.cuda()  # Move image batch to GPU
        labels = torch.LongTensor(labels)
        outputs = model(images)
        _, predicted = outputs.data
        total += len(labels)
        correct += (predicted.cpu() == labels).sum()
        if using_gpu:  # GPU cache cleaning. Unsure of effectiveness
            torch.cuda.empty_cache()
    if task == "cross-validate":
        accuracy['cval'].append(100 * correct / total)  # To plot
        return total, accuracy['cval'][-1]
    elif task == "test":
        accuracy['test'].append(100 * correct / total)  # To plot
        model.results['accuracy'] = accuracy['test'][-1]
        return total
