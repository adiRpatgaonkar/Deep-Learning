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
    
    # Parse argments provided
    do.parse_arg()
    # Setup GPU or CPU
    init_setup.setup_hardware()
    
    global model
    # Load or create new ?
    if do.args.LOAD:
        print('\nWorking with loaded model.\n')         
        if do.args.FIT:
            model = nnc.load_model(do.args.LOAD)
            print('Fitting net for loaded model')
            model, fitting_loader = fitting_net.fit(model)
            if do.args.TEST:
                print('Testing model fitting:')
                test_net.test(model, fitting_loader)
            if do.args.INFER:
                print('Inferencing model fitting:')
                infer.inferences(model)
            do.args.FIT = False
        elif do.args.TRAIN:
            model = nnc.load_model(do.args.LOAD)
            print('Training net for loaded model')
            model = train_net.train(model)
            if do.args.TEST:
                print('Testing trained model:')
                test_net.test(model)
            if do.args.INFER:
                print('Inferencing trained model:')
                infer.inferences(model)
            do.args.TRAIN = False
        elif do.args.TEST:
            model = nnc.load_model(do.args.LOAD)
            print('Testing net for loaded model')
            test_net.test(model)
        elif do.args.INFER:
            model = nnc.load_model(do.args.LOAD)            
            print('Testing net for loaded model')
            infer.inferences(model)
                
    elif do.args.NEW:
        print('\nWorking with new model.\n')
        if do.args.FIT:
            print('Fitting net for new model')
            model, fitting_loader = fitting_net.fit()
            if do.args.TEST:
                print('Testing model fitting:')
                test_net.test(model, fitting_loader)
            if do.args.INFER:
                print('Inferencing model fitting:')
                infer.inferences(model)
            do.args.FIT = False
        elif do.args.TRAIN:
            print('Training net for new model')
            model = train_net.train()
            if do.args.TEST:
                print('Testing trained model:')
                test_net.test(model)
            if do.args.INFER:
                print('Inferencing trained model:')
                infer.inferences(model)        
            do.args.TRAIN = False

    # Final goodbye
    print('\n' + '-' * 7 + '\nExiting\n' + '-' * 7)
    # Clear cache, just to be sure. Unsure of function effectivity
    if do.use_gpu:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
