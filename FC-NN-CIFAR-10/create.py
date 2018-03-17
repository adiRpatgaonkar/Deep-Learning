""" Create stuff """

from __future__ import print_function
import yaml

from do_stuff import arguments
import nnCustom as nnc


def create_model():
    global args
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


def set_hyper_parameters(config, model):
    global cfg
    with open(config, 'r') as f:
        cfg = yaml.load(f)

    model.model_type += cfg["MODEL"]["TYPE"]

    model.weights_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
    model.reg = cfg["SOLVER"]["REG"]
    if args.FIT:
        model.data_set = cfg["FIT"]["DATASET"]
        model.lr = cfg["FIT"]["BASE_LR"]
        model.lr_policy += cfg["FIT"]["LR_POLICY"]
        model.decay_rate = cfg["FIT"]["DECAY_RATE"]
        model.epochs = cfg["FIT"]["EPOCHS"]
    elif args.TRAIN:
        model.data_set = cfg["TRAIN"]["DATASET"]
        model.lr = cfg["TRAIN"]["BASE_LR"]
        model.lr_policy += cfg["TRAIN"]["LR_POLICY"]
        model.decay_rate = cfg["TRAIN"]["DECAY_RATE"]
        model.epochs = cfg["TRAIN"]["EPOCHS"]
    if args.TEST:
        model.data_set = cfg["TEST"]["DATASET"]
    if args.INFER:
        model.data_set = cfg["TEST"]["DATASET"]
    return


def configs():
    return cfg
