from __future__ import print_function
from termcolor import colored
import yaml
import torch
import numpy as np

import do_stuff as do
import nnCustom as nnc
import Dataset as dset
import create

def fit(model=None):

    if model is None:
        model = create.create_model()
          
    # Model fitting test
    print("\n+++++     Model fitting     +++++\n")
    # Get data
    train_dataset = dset.CIFAR10(directory='data', download=True, train=True)  
    optimizer = nnc.Optimize(model)
    print("Learning rate: %.4f\n" % model.lr)
    fitting_loader = dset.data_loader(train_dataset.data, batch_size=dset.CIFAR10.batch_size, model_testing=True)
    for epoch in range(model.epochs):
        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        for images, labels in fitting_loader:
            if do.use_gpu:
                images = images.cuda()
            # print(type(images))
            model.train(images, labels)
            if do.use_gpu:
                torch.cuda.empty_cache()
        print(colored('# Fitting test Loss:', 'red'), end="")
        print('[%.4f] @ L.R: %.9f' % (model.loss, model.lr))
        model.loss_history.append(model.loss)
        optimizer.time_decay(epoch, 0.0005)
        optimizer.set_optim_param(epoch)
        
    model.plot_loss()
        
    # Model status
    model.model_fitted = model.optimum['Fitting tested'] = True        
    print("\nModel status:")
    print("{ Fitting tested:", model.optimum['Fitting tested'], "|", "Trained:", model.optimum['Trained'], "|", 
          "Tested:", model.optimum['Tested'], "|", "Inferenced:", model.optimum['Inferenced'], "}\n")
    print("{ Loss:", model.optimum['Loss'], "}\n")
    
    model.set_logs()    
    # Saving fitted model    
    if do.args.SAVE:
        nnc.save_model('model.pkl', model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            nnc.save_model('model.pkl', model)
        else:
            print('Not saving model.')                                
        
    return model, fitting_loader