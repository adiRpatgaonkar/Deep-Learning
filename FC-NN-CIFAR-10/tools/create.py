""" Create stuff """

from __future__ import print_function

from configs.config_model import set_hyper_parameters
from libs import nn as nnc
from libs.check_args import arguments


def create_model():
    args = arguments()
    # Define the network
    print('\n' + '+' * 20, '\nBuilding net & model\n' + '+' * 20)
    model = nnc.ModelNN()
    set_hyper_parameters(args.CFG, model)
    model.add(nnc.LinearLayer(32 * 32 * 3, 1024))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(1024, 512))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(512, 10))
    model.add(nnc.CeCriterion('Softmax'))
    return model
