"""
How to use the model?

@author: apatgao
"""

import sys

using_gpu, model_load, model_fit, model_train, model_test = False, False, False, False, False
gpu_id = mfit_epochs = train_epochs = 0
if len(sys.argv) > 1:
    if '--gpu' in sys.argv:
        using_gpu = True
        gpu_id = int(sys.argv[sys.argv.index("--gpu") + 1])
    if '--load' in sys.argv:
        model_load = True        
    if '--fit' in sys.argv:
        model_fit = True
        mfit_epochs = int(sys.argv[sys.argv.index("--fit") + 1])        
    if '--train' in sys.argv:
        model_train = True
        train_epochs = int(sys.argv[sys.argv.index("--train") + 1])        
    if '--test' in sys.argv:
        model_test = True
else:
    print('Usage: python main.py [--gpu] [--load] [--fit] [--train] [--test]')
