""" Create stuff """

import do_stuff as do
import nnCustom as nnc


def create_model():
    # Define the network
    print('\n' + '+' * 16, '\nDefining network\n' + '+' * 16)
    global model
    model = nnc.ModelNN()
    set_hyper_paramters(do.args.CFG)
    model.add(nnc.LinearLayer(32 * 32 * 3, 1024))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(1024, 512))
    model.add(nnc.Activation('ReLU'))
    model.add(nnc.LinearLayer(512, 10))
    model.add(nnc.CeCriterion('Softmax'))
    return model
