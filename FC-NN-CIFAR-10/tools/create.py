""" Create stuff """

# System imports
from __future__ import print_function

# Custom imports
from configs.config_model import set_hyper_parameters
from church import nn
from libs.check_args import arguments


def create_model():
    """ Build the net & model """
    args = arguments()
    # Define the network
    print('\n' + '+' * 20, '\nBuilding net & model\n' + '+' * 20)
    
    model = nn.ModelNN()

    set_hyper_parameters(args.CFG, model)

    model.add(nn.Linear(32 * 32 * 3, 2048))
    model.add(nn.Activation('ReLU'))
    model.add(nn.Linear(2048, 512))
    model.add(nn.Activation('ReLU'))
    model.add(nn.Linear(512, 128))
    model.add(nn.Activation('ReLU'))
    model.add(nn.Linear(128, 10))
    model.add(nn.CeCriterion('Softmax'))

    return model
