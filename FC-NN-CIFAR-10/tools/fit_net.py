""" Fitting code for new/saved models """

# System imports
from __future__ import print_function

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
    train_dataset = CIFAR10(directory='data', download=True, train=True)
    # Optimizer  
    optimizer = Optimize(model)
    print("\n# Stochastic gradient descent #")
    print("Learning rate: %.4f\n" % model.lr)
    fitting_loader = data_loader(train_dataset.data, batch_size=CIFAR10.batch_size, model_testing=True)
    for epoch in range(model.epochs):
        print('Epoch: [%d/%d]' % (epoch + 1, model.epochs), end=" ")
        for images, labels in fitting_loader:
            if using_gpu():
                images = images.cuda()
            # print(type(images))
            model.train(images, labels)
            if using_gpu():
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
    if args.SAVE:
        save_model(args.SAVE, model)
    else:
        f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
        if f.lower() == 'y' or f.lower() == 'yes':
            save_model('model.pkl', model)
        else:
            print('Not saving model.')

    return [model, fitting_loader]
