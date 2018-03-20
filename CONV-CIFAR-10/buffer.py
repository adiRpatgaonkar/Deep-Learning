#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:42:29 2018

@author: apatgao
"""
from __future__ import print_function
import nnCustom as nnc
import numpy as np

convModel = nnc.ModelNN()

# image = np.ones((1, 3, 227, 227))
# convModel.add(nnc.Conv2D(image.shape[1:], f=11, k=96, pad=0, stride=4))
# image = np.ones((1, 3, 5, 5))
# convModel.add(nnc.Conv2D(image.shape[1:], f=3, k=2, pad=1, stride=2))

image = np.random.randint(256, size=(1, 3, 32, 32))

convModel.add(nnc.Conv2D([3, 32, 32], f=5, k=6))
feature_maps = convModel.layers[0].convolve(image)
print("Output volume", feature_maps.size())

convModel.add(nnc.SpatialPool2D(feature_maps.size(), 2, 2, 'max'))
pooled_maps = convModel.layers[1].pooling(feature_maps)
print("Pooled volume", pooled_maps.size())

convModel.add(nnc.Conv2D([6, 14, 14], f=5, k=16))
feature_maps = convModel.layers[2].convolve(pooled_maps.unsqueeze(0))
print("Output volume", feature_maps.size())

convModel.add(nnc.SpatialPool2D(feature_maps.size(), 2, 2, 'max'))
pooled_maps = convModel.layers[3].pooling(feature_maps)
print("Pooled volume", pooled_maps.size())

convModel.add(nnc.Linear(32 * 32 * 3, 1024))
convModel.add(nnc.Activation('ReLU'))
convModel.add(nnc.Linear(1024, 512))
convModel.add(nnc.Activation('ReLU'))
convModel.add(nnc.Linear(512, 10))
convModel.add(nnc.Criterion('Softmax'))
convModel.show_net()
