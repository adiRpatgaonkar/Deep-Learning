import do_stuff as do
import nnCustom as nnc
import fitting_net
import train_net
import test_net
import infer
import create

def main():
    global model
    do.parse_arg()

    if do.args.LOAD:
        model = create.create_model()
        model = nnc.load_model(do.args.LOAD, model)

    if do.args.FIT:
        model, fitting_loader = fitting_net.fit()

    if do.args.TRAIN:
        model = train_net.train()

    if do.args.TEST:
        if do.args.FIT:      
            test_net.test(model, fitting_loader)
        else:
            test_net.test(model)

    if do.args.INFER:
        if do.args.FIT:
            infer.inferences(model, fitting_loader)
        else:
            infer.inferences(model)

    print('\n' + '-' * 7 + '\nExiting\n' + '-' * 7)
    if do.args.GPU:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()