import torch

import do_stuff as do
import nnCustom as nnc
import fitting_net
import train_net
import test_net
import infer
import create
import init_setup

def main():
    
    do.parse_arg()
    
    init_setup.setup_hardware()
    
    global model
    if do.args.LOAD:
        print('\nWorking with loaded model.\n')         
        if do.args.FIT:
            fitting = True
            model = create.create_model()
            model = nnc.load_model(do.args.LOAD, model)
            print('Fitting net for loaded model')
            model, fitting_loader = fitting_net.fit(model)
            if do.args.TEST:
                print('Testing model fitting:')
                test_net.test(model, fitting_loader)
            if do.args.INFER:
                print('Inferencing model fitting:')
                infer.inferences(model)
            do.args.FIT = False
        if do.args.TRAIN:
            model = create.create_model()
            model = nnc.load_model(do.args.LOAD, model)
            print('Training net for loaded model')
            model = train_net.train(fitting, model)
            if do.args.TEST:
                print('Testing trained model:')
                test_net.test(model)
            if do.args.INFER:
                print('Inferencing trained model:')
                infer.inferences(model)
            do.args.TRAIN = False
                
    elif do.args.NEW:
        print('\nWorking with new model.\n')
        if do.args.FIT:
            fitting = True
            print('Fitting net for new model')
            model, fitting_loader = fitting_net.fit()
            if do.args.TEST:
                print('Testing model fitting:')
                test_net.test(model, fitting_loader)
            if do.args.INFER:
                print('Inferencing model fitting:')
                infer.inferences(model)
            do.args.FIT = False
        if do.args.TRAIN:
            print('Training net for new model')
            model = train_net.train(fitting)
            if do.args.TEST:
                print('Testing trained model:')
                test_net.test(model)
            if do.args.INFER:
                print('Inferencing trained model:')
                infer.inferences(model)        
            do.args.TRAIN = False

    print('\n' + '-' * 7 + '\nExiting\n' + '-' * 7)
    if do.use_gpu:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()