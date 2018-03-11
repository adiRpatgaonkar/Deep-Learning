import do_stuff as do
import nnCustom as nnc
import fitting_net
import train_net
import test_net
import infer
import create

def main():

    do.parse_arg()

    if do.args.LOAD:
        model = nnc.load_model(do.args.LOAD)

    if do.args.FIT:
        fitting_net.fit()

    if do.args.TRAIN:
        model = train_net.train()

    if do.args.TEST:
        test_net.test(model)

    if do.args.INFER:
        if not do.args.TRAIN:
            model = create.create_model()
        infer.inferences(model)

    print('\n' + '-' * 7 + '\nExiting\n' + '-' * 7)
    if do.args.GPU:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()