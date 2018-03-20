""" Starts the drama """

# System imports
import torch

# Custom imports
from libs.check_args import using_gpu, parse_arg, arguments
from libs.setup import setup_hardware

from tools.model_store import load_model
from tools.fit_net import fit
from tools.train_net import train
from tools.test_net import test
from tools.infer import inferences
from tools.best_model import best_model_selection

def main():
    
    # Parse arguments provided
    parse_arg()
    args = arguments()
    # Setup GPU or CPU
    setup_hardware()
    
    global model
    # Load or create new ?
    if args.bms:
        best_model_selection(replace=True)
    elif args.LOAD:
        print('\nWorking with loaded model.')
        if args.FIT:
            model = load_model(args.LOAD)
            print('Fitting net for loaded model')
            model, fitting_loader = fit(model)
            if args.TEST:
                print('Testing model fitting:')
                test(model, fitting_loader)
            if args.INFER:
                print('Inference model fitting:')
                inferences(model)
            args.FIT = False
        elif args.TRAIN:
            model = load_model(args.LOAD)
            print('Training net for loaded model')
            model = train(model)
            if args.TEST:
                print('Testing trained model:')
                test(model)
            if args.INFER:
                print('Inference trained model:')
                inferences(model)
            args.TRAIN = False
        elif args.TEST:
            model = load_model(args.LOAD)
            print('Testing net for loaded model')
            test(model)
        elif args.INFER:
            model = load_model(args.LOAD)
            print('Testing net for loaded model')
            inferences(model)
                
    elif args.NEW:
        print('\nWorking with new model.')
        if args.FIT:
            print('Fitting net for new model')
            model, fitting_loader = fit()
            if args.TEST:
                print('Testing model fitting:')
                test(model, fitting_loader)
            if args.INFER:
                print('Inference model fitting:')
                inferences(model)
            args.FIT = False
        elif args.TRAIN:
            print('Training net for new model')
            model = train()
            if args.TEST:
                print('Testing trained model:')
                test(model)
            if args.INFER:
                print('Inference trained model:')
                inferences(model)
            args.TRAIN = False

    # Final goodbye
    print('\n' + '-' * 7 + '\nExiting\n' + '-' * 7)
    # Clear cache if using GPU (Unsure of effectiveness)
    if using_gpu():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

