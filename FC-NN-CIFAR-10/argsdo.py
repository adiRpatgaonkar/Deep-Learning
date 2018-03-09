"""
How to use the model?

@author: apatgao
"""

import sys

using_gpu, model_load, model_fit, model_train, model_test = False, False, False, False, False
gpu_id = mfit_epochs = train_epochs = 0
if len(sys.argv) > 1:
    if '--GPU' in sys.argv:
        using_gpu = True
        gpu_id = int(sys.argv[sys.argv.index("--GPU") + 1])
    if '--LOAD' in sys.argv:
        model_load = True        
    if '--FIT' in sys.argv:
        model_fit = True
        mfit_epochs = int(sys.argv[sys.argv.index("--FIT") + 1])        
    if '--TRAIN' in sys.argv:
        model_train = True
        train_epochs = int(sys.argv[sys.argv.index("--TRAIN") + 1])        
    if '--TEST' in sys.argv:
        model_test = True
else:
    print('Usage: python main.py [--GPU] [--LOAD] [--FIT] [--TRAIN] [--TEST]')
