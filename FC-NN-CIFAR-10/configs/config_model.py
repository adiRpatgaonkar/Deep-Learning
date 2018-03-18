""" Model configuration setup """

# System imports
import yaml

# Custom imports
from libs.check_args import arguments


def set_hyper_parameters(config_file, model):
    global cfg
    args = arguments()
    with open('configs/' + config_file, 'r') as f:
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
