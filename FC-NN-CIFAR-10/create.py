""" Create stuff """

from __future__ import print_function
import yaml

import do_stuff as do
import nnCustom as nnc


def create_model():
    # Define the network
    print('\n' + '+' * 16, '\nDefining network\n' + '+' * 16)
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

    with open(config, 'r') as f:
        cfg = yaml.load(f)

    model.type += cfg["MODEL"]["TYPE"]
   
    model.weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
    model.reg = cfg["SOLVER"]["REG"]
    if do.args.FIT:
        model.lr = cfg["FIT"]["BASE_LR"]
        model.lr_policy += cfg["FIT"]["LR_POLICY"]
        model.decay_rate = cfg["FIT"]["DECAY_RATE"]
        model.epochs = cfg["FIT"]["EPOCHS"]
        return
    if do.args.TRAIN:
        model.lr = cfg["TRAIN"]["BASE_LR"]
        model.lr_policy += cfg["TRAIN"]["LR_POLICY"]
        model.decay_rate = cfg["TRAIN"]["DECAY_RATE"]
        model.epochs = cfg["TRAIN"]["EPOCHS"]
        return