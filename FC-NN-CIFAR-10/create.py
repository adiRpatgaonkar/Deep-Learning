""" Create stuff """

from __future__ import print_function
import yaml

import do_stuff as do
import nnCustom as nnc


def create_model():
    # Define the network
    print('\n' + '+' * 20, '\nBuilding net & model\n' + '+' * 20)
    global model
    model = nnc.ModelNN()
    set_hyper_paramters(do.args.CFG, model)
    model.add(nnc.LinearLayer(32 * 32 * 3, 1024))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(1024, 512))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(512, 10))
    model.add(nnc.CeCriterion('Softmax'))
    return model

def set_hyper_paramters(config, model):
    global cfg
    with open(config, 'r') as f:
        cfg = yaml.load(f)

    model.model_type += cfg["MODEL"]["TYPE"]
   
    model.weights_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
    model.reg = cfg["SOLVER"]["REG"]
    if do.args.FIT:
        model.lr = cfg["FIT"]["BASE_LR"]
        model.lr_policy += cfg["FIT"]["LR_POLICY"]
        model.decay_rate = cfg["FIT"]["DECAY_RATE"]
        model.epochs = cfg["FIT"]["EPOCHS"]
    elif do.args.TRAIN:
        model.lr = cfg["TRAIN"]["BASE_LR"]
        model.lr_policy += cfg["TRAIN"]["LR_POLICY"]
        model.decay_rate = cfg["TRAIN"]["DECAY_RATE"]
        model.epochs = cfg["TRAIN"]["EPOCHS"]
    return