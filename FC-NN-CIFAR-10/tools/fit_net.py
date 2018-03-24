""" Fitting code for new/saved models """

# System imports
from __future__ import print_function
from termcolor import colored
# Custom imports
from data.dataset import CIFAR10, data_loader
from libs.check_args import arguments, using_gpu
from libs.nn import Optimize
from model_store import save_model
import create


# Model fitting test
def fit(model=None):

    args = arguments()

    if model is None:
        model = create.create_model()

    print("\n+++++     FITTING     +++++\n")
    model.show_log(arch=True, fit=True)

    # Get data
    train_dataset = CIFAR10(directory='data', 
        download=True, 
        train=True)

    # Optimizer/Scheduler
    optimizer = Optimize(model)

    # SGD
    print("\n# Stochastic gradient descent #")
    print("Learning rate: %.4f\n" % model.lr)

    # Get one batch from the dataset
    fitting_loader = data_loader(data=train_dataset.data, 
        batch_size=CIFAR10.batch_size,
        fit_testing=True)

    # Epochs
    for epoch in range(model.epochs):
        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        for images, labels in fitting_loader:
            if using_gpu():
                images = images.cuda()
            model.train(images, labels)
            # Clear cache if using GPU (Unsure of effectiveness)
            if using_gpu():
                torch.cuda.empty_cache()
        # Print fitting loss
        print(colored('# Fitting test Loss:', 'red'), end="")
        print('[%.4f] @ L.R: %.9f' % (model.train_loss, model.lr))
        model.train_loss_history.append(model.train_loss)

        optimizer.time_decay(epoch, 0.0005)
        optimizer.set_optim_param(epoch)

    # Plot fitting history   
    model.plot_history(loss_history=True)
    # Model status
    model.fitted = True

    model.show_log(curr_status=True)
    model.set_logs()

    # Saving fitted model    
    if args.SAVE:
        save_model(args.SAVE, model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            save_model('model.pkl', model)
        else:
            print('Not saving model.')

    return [model, fitting_loader]