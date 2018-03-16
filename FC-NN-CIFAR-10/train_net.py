from __future__ import print_function
from termcolor import colored
import yaml
import torch


import do_stuff as do
import nnCustom as nnc
import Dataset as dset
import create
# Training
def train(model=None):

    if model is None:
        model = create.create_model()
    
    print("\n+++++     TRAINING     +++++\n")
    model.show_log(arch=True, train=True)
    # Get data
    train_dataset = dset.CIFAR10(directory='data', download=True, train=True)
    # Optimizer
    optimizer = nnc.Optimize(model)
    print("\n# Stochastic gradient descent #")
    print("Learning rate: %.4f\n" % model.lr)
    for epoch in range(model.epochs):
        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        train_loader = dset.data_loader(train_dataset.data, batch_size=dset.CIFAR10.batch_size, shuffled=True)
        for images, labels in train_loader:
            if do.use_gpu:
                images = images.cuda()
            model.train(images, labels)
            if do.use_gpu:
                torch.cuda.empty_cache()
        print(colored('# Training Loss:', 'red'), end=" ")
        print('[%.4f] @ L.R: %.4f' % (model.loss, model.lr))
        model.loss_history.append(model.loss)
        optimizer.time_decay(epoch, 0.005)
        optimizer.set_optim_param(epoch)
    
    model.plot_loss()
    
    # Model status
    model.model_trained = model.optimum['Trained'] = True        
    print("\nModel status:")
    print("{ Fitting tested:", model.optimum['Fitting tested'], "|", "Trained:", model.optimum['Trained'], "|", 
          "Tested:", model.optimum['Tested'], "|", "Inferenced:", model.optimum['Inferenced'], "}")
    print("{ Loss:", model.optimum['Loss'], "}\n")
    
    model.set_logs()        
    # Saving fitted model    
    if do.args.SAVE:
        nnc.save_model(do.args.SAVE, model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            nnc.save_model('model.pkl', model)
        else:
            print('Not saving model.')

    return model
