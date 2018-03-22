""" Create stuff """

# System imports
from __future__ import print_function

# Custom imports
from configs.config_model import set_hyper_parameters
from libs import nn as nnc
from libs.check_args import arguments


def create_model():
    """ Build the net & model """
    args = arguments()
    # Define the network
    print('\n' + '+' * 20, '\nBuilding net & model\n' + '+' * 20)
    
    model = nnc.ModelNN()

    set_hyper_parameters(args.CFG, model)

    model.add(nnc.LinearLayer(32 * 32 * 3, 2048))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(2048, 512))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(512, 128))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(128, 10))
    model.add(nnc.CeCriterion('Softmax'))

    return model
