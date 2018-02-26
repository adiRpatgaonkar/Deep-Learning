#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
How to use the model?

Created on Fri Feb 16 00:11:58 2018

@author: apatgao
"""

import sys

using_gpu, model_load, model_fit, model_train, model_test = False, False, False, False, False 
if len(sys.argv) > 1:
    if '--GPU' in sys.argv:
        using_gpu = True
    if '--LOAD' in sys.argv:
        model_load = True
    if '--FIT' in sys.argv:
        model_fit = True
    if '--TRAIN' in sys.argv:
        model_train = True
    if '--TEST' in sys.argv:
        model_test = True
else:
    print('Usage: python main.py [--GPU] [--LOAD] [--FIT] [--TRAIN] [--TEST]')
