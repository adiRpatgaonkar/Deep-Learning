#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Custom sequential neural network framework

Created on torch Feb  8 17:51:22 2018

@author: apatgao
"""
from __future__ import print_function
import numpy as np, torch, matplotlib.pyplot as plt, math, pickle, argsdo as do
import Dataset as dset

if do.using_gpu and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def feature_scale(data, data_size, batch_norm=False):
    """ Normalizes the given data with mean and standard deviation """
    if batch_norm:
        print('Not implemented.')
    else:
        data = data.view(data_size, -1)  # flatten
        mean = torch.mean(data, 1, keepdim=True)
        std_deviation = torch.std(data, 1, keepdim=True)  # print mean, std_deviation
        data = data - mean
        data = data / std_deviation
    return data


def save_model(filename, nn_model):
    f = raw_input('Do you want to save the model? (y)es/(n)o: ').lower()
    if f.lower() == 'y' or f.lower() == 'yes':
        f = open(filename, 'wb')
        print('\nSaving ...', end=" ")
        pickle.dump(nn_model.optimum, f)
        print('done.')
        f.close()
    else:
        print('Not saving model.')


def load_model(filename, nn_model=None):
    if nn_model is None:
        print('\nChecking saved models ...')
        return pickle.load(open(filename, 'rb'))
    print('\nLoading model from %s ...' % filename)
    t = pickle.load(open(filename, 'rb'))
    i = 0
    for layer in nn_model.layers:
        if layer.LayerName == 'Linear':
            layer.w = t['Weights'][i]
            layer.b = t['Biases'][i]
            i += 1


class ModelNN(object):
    """
    Model class encapsulating all layers, functions, hyper parameters etc.
    1. Train
    2. Test
    3. Fprop & Backprop
    4. Cross entropy loss
    5. Parameter updates
    6. Plot loss
    7. Display results
    """

    def __init__(self):
        """
        :rtype: object
        """
        self.net, self.layers, self.loss_history = "", [], []
        self.num_layers = 0
        self.weights, self.biases, self.output = [], [], []
        self.grad_weights, self.grad_biases, self.grad_output = [], [], []
        self.epochs = self.lr = self.decay_rate = 1
        self.reg = 1e-3  # regularization strength
        self.loss = self.predictions = self.train_acc = self.test_acc = 0
        self.optimum = {'Net': "", 'Loss': 10, 'Epoch': 0, 'Learning rate': self.lr, 'Weights': 0,
                        'Biases': 0, 'TrainAcc': self.train_acc, 'TestAcc': self.test_acc}
        self.isTrain = False

    def add(self, lyrObj):
        """ Add layers, activations to the nn architecture """
        self.layers.append(lyrObj)
        if lyrObj.LayerName == 'Linear':
            self.weights.append(lyrObj.w)
            self.biases.append(lyrObj.b)
            self.grad_weights.append(0)
            self.grad_biases.append(0)
        self.output.append(0)
        self.grad_output.append(0)
        self.num_layers += 1

    def show_net(self):
        """ Display the network """
        print('\nNet arch:', end=" ")
        self.net += '{\n'
        for i, l in enumerate(self.layers):
            self.net += str(i) + ': ' + l.LayerName
            if l.LayerName == 'Convolution 2D':
                self.net += '(' + str(l.kernels.size(2)) + 'x' + str(l.kernels.size(3)) + ', ' + str(l.filters) + ')'
            elif l.LayerName == 'Pooling 2D':
                self.net += '(' + l.pool_type + ': ' + str(l.spatial_extent) + 'x' + str(l.spatial_extent) + ')'
            elif l.LayerName == 'Linear':
                self.net += '(' + str(l.ipt_neurons) + 'x' + str(l.opt_neurons) + ')'
            elif l.LayerName == 'Activation':
                self.net += '(' + l.activation + ')'
            elif l.LayerName == 'Criterion':
                self.net += '(' + l.classifier + ')'
            self.net += '-->\n'
        self.net += '}'
        self.optimum['Net'] = self.net

        print(self.net)

    def train(self, ipt, label):
        """ Fprop and Backprop to train """
        self.isTrain = True
        ipt = feature_scale(ipt, ipt.size(0))
        self.forward(ipt, label)
        self.backward(ipt, label)

    def test(self, ipt, target):
        """ Fprop to test the model """
        self.isTrain = False
        self.forward(feature_scale(ipt, ipt.size(0)), target)

    def forward(self, ipt, label):
        """ Feedforward for sequential NN layers """
        for lth in range(self.num_layers):
            if lth == 0:  # Input layer
                if self.layers[lth].LayerName == 'Linear':
                    self.output[lth] = self.layers[lth].forward(ipt)
            elif lth < self.num_layers - 1:  # Hidden layers
                if self.layers[lth].LayerName == 'Linear':
                    self.output[lth] = self.layers[lth].forward(self.output[lth - 1])
                elif self.layers[lth].LayerName == 'Activation':
                    if self.layers[lth].activation == 'ReLU':
                        self.output[lth] = self.layers[lth].relu(self.output[lth - 1])
            else:  # Last layer
                if self.isTrain:
                    if self.layers[lth].LayerName == 'Criterion':
                        if self.layers[lth].classifier == 'Softmax':
                            _, _, self.output[lth] = self.layers[lth].softmax(self.output[lth - 1])
                else:
                    if self.layers[lth].LayerName == 'Criterion':
                        self.output[lth], self.predictions, _ = \
                            (self.layers[lth].softmax(self.output[lth - 1], label))
                self.cross_entropy_loss(self.output[lth], label)

    def backward(self, ipts, targets):
        """ Backpropogation for sequential NN layers """
        param = len(self.weights) - 1
        for lth in range(self.num_layers - 1, -1, -1):
            if lth == self.num_layers - 1:  # Last layer
                if self.layers[lth].LayerName == 'Criterion':
                    if self.layers[lth].classifier == 'Softmax':
                        self.grad_output[lth] = self.layers[lth].backward_softmax(self.output[lth], targets)
            elif self.num_layers - 1 > lth > 0:  # Hidden layers
                if self.layers[lth].LayerName == 'Linear':
                    self.grad_weights[param], self.grad_biases[param] = (
                        self.layers[lth].backward(self.output[lth - 1], self.grad_output[lth + 1]))
                    self.grad_output[lth] = \
                        (self.layers[lth].backward(self.weights[param], self.grad_output[lth + 1], 1))
                    param -= 1
                elif self.layers[lth].LayerName == 'Activation':
                    if self.layers[lth].activation == 'ReLU':
                        self.grad_output[lth] = \
                            (self.layers[lth].backward_relu(self.output[lth], self.grad_output[lth + 1]))
            else:
                self.grad_weights[0], self.grad_biases[0] = \
                    (self.layers[0].backward(ipts, self.grad_output[1]))
        self.update_parameters()

    def cross_entropy_loss(self, softmax, targets):
        """ Cross entropy loss """
        if self.isTrain:
            correct_log_probs = (-(torch.log(softmax[range(dset.CIFAR10.batch_size), targets])
                                   / torch.log(torch.Tensor([10]).type(dtype))))
            # print correct_log_probs
            self.loss = torch.sum(correct_log_probs) / dset.CIFAR10.batch_size
            weights = self.parameters()
            reg_loss = 0
            for w in weights[0]:
                reg_loss += 0.5 * self.reg * torch.sum(w * w)
            self.loss += reg_loss
        else:
            probs = -(torch.log(softmax) / torch.log(torch.Tensor([10]).type(dtype)))
            self.loss = torch.sum(probs) / dset.CIFAR10.test_size

    def update_parameters(self):
        """ Bias and weight updates """
        for i, (grad_ws, grad_bs) in enumerate(zip(self.grad_weights, self.grad_biases)):
            grad_ws += self.reg * self.weights[i]
            self.weights[i] += (-self.lr * grad_ws)
            self.biases[i] += (-self.lr * grad_bs)

    def parameters(self):
        """ Returns parameters """
        return [self.weights, self.biases]

    def plot_loss(self, to_show=False):
        """ Plot gradient descent curve """
        plt.plot(range(len(self.loss_history)), self.loss_history, linewidth=2.1)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if to_show:
            plt.show()

    def display_results(self, target, test_set=None, all_exp=False):
        """ Display model results i.e. predictions on test set """
        if all_exp:
            for example in range(dset.CIFAR10.test_size):
                print("Ground truth: (%d) %s || Prediction: (%d) %s || Confidence: %.2f %" %
                      (target[example], dset.CIFAR10.classes[int(target[example])],
                       int(self.predictions[example]),
                       dset.CIFAR10.classes[int(self.predictions[example])],
                       self.output[-1][example] * 100))
        else:
            test_set = test_set.cpu()
            test_set = \
                (test_set.numpy().reshape(dset.CIFAR10.test_size, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8"))
            while True:
                example = input("Which test example? (0-9999): ")
                if example < 0 or example >= dset.CIFAR10.test_size:
                    return
                print('Ground truth: (%d) %s' % (int(target[example]),
                                                 dset.CIFAR10.classes[int(target[example])]))
                plt.imshow(test_set[example])
                plt.xlabel(str(int(self.predictions[example])) + ' : ' +
                           dset.CIFAR10.classes[int(self.predictions[example])])
                plt.ylabel('Confidence: ' + str(format(self.output[-1][example] * 100, '.2f')) + '%')
                plt.show()


class Conv2D(ModelNN):
    LayerName = 'Convolution 2D'

    def __init__(self, input_dim, f, k, pad=0, stride=1):
        super(Conv2D, self).__init__()
        self.filters, self.padding, self.strides = k, pad, stride
        self.depth, self.height, self.width = input_dim[0:]
        self.output_dim = [0, 0, 0]
        self.output_dim[0] = self.filters
        self.output_dim[1] = ((self.width - f + 2 * pad) / stride) + 1
        self.output_dim[2] = ((self.height - f + 2 * pad) / stride) + 1
        self.feature_volume = torch.zeros(0, 0)
        self.kernels = 0.01 * torch.randn(self.filters, self.depth, f, f)  # print(self.kernels)
        self.biases = torch.ones(self.filters, 1, 1, 1)  # print(self.biases)
        # print(self.output_dim, self.feature_volume.size(), self.kernels.size())

    def convolve(self, images):

        if type(images) is torch.FloatTensor:
            images = images.numpy()
        images = np.pad(images, pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                        mode='constant', constant_values=0)
        images = torch.from_numpy(images).type(torch.FloatTensor)
        fh = fw = self.kernels.size(2)
        for image in images:
            for bias, kernel in zip(self.biases, self.kernels):  # print(kernel, bias)
                temp = []
                for i in range(0, image.size(1) - kernel.size(1) + 1, self.strides):
                    for j in range(0, image.size(1) - kernel.size(2) + 1, self.strides):
                        # print(torch.sum(image[:, i:fh, j:fw] * kernel + bias))
                        temp.append(torch.sum(image[:, i:fh, j:fw] * kernel) + bias)
                        fw += self.strides
                    fh += self.strides
                    fw = self.kernels.size(2)
                fh = self.kernels.size(2)
                temp = torch.from_numpy(np.asarray(temp, dtype='float32'))
                o1 = temp.clone()
                o1.resize_(self.output_dim[1], self.output_dim[2])
                self.feature_volume = torch.cat((self.feature_volume, o1), dim=0)
        self.feature_volume.resize_(self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return self.feature_volume


class SpatialPool2D(ModelNN):
    LayerName = 'Pooling 2D'

    def __init__(self, input_dim, f, stride, pool='max'):
        super(SpatialPool2D, self).__init__()
        self.pool_type = pool
        self.strides = stride
        self.spatial_extent = f
        self.height, self.width = input_dim[1:]
        self.output_dim = [0, 0, 0]
        self.output_dim[0] = input_dim[0]
        self.output_dim[1] = ((self.width - f) / stride) + 1
        self.output_dim[2] = ((self.height - f) / stride) + 1
        self.pooled_volume = torch.zeros(0, 0)

    def pooling(self, feature_maps=None):

        fh = fw = self.spatial_extent
        for a_map in feature_maps:
            temp = []
            for i in range(0, a_map.size(1) - self.spatial_extent + 1, self.strides):
                for j in range(0, a_map.size(1) - self.spatial_extent + 1, self.strides):
                    if self.pool_type == 'max':
                        temp.append(torch.max(a_map[i:fh, j:fw]))
                    elif self.pool_type == 'mean':
                        temp.append(torch.mean(a_map[i:fh, j:fw]))
                    fw += self.strides
                fh += self.strides
                fw = self.spatial_extent
            fh = self.spatial_extent
            temp = torch.from_numpy(np.asarray(temp, dtype='float32'))
            o1 = temp.clone()
            o1.resize_(self.output_dim[1], self.output_dim[2])
            self.pooled_volume = torch.cat((self.pooled_volume, o1), dim=0)
        self.pooled_volume.resize_(self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return self.pooled_volume


class Linear(ModelNN):
    """
    Linear Layer class
    Fully connected layer
    """

    LayerName = 'Linear'

    def __init__(self, num_ipt_neurons, num_opt_neurons):
        # print 'Linear layer created'
        # allocate size for torche state variables appropriately
        super(Linear, self).__init__()
        self.ipt_neurons = num_ipt_neurons
        self.opt_neurons = num_opt_neurons
        self.w = 0.01 * torch.rand(num_ipt_neurons, num_opt_neurons).type(dtype)
        self.b = torch.zeros(1, num_opt_neurons).type(dtype)

    def forward(self, ipt, target=None):
        #  if not dset.isTrain:
        #    print('w & b @ Linear layer: ', self.w)
        # print 'I/P @ Linear layer:', ipt.size()
        output = torch.mm(ipt, self.w) + self.b
        return output

    def backward(self, ipt, grad_output, i=-1):
        if i == -1:
            grad_w = torch.mm(ipt.t(), grad_output)
            grad_b = torch.sum(grad_output, dim=0, keepdim=True)
            return [grad_w, grad_b]
        grad_output = torch.mm(grad_output, ipt.t())
        return grad_output


class Activation(ModelNN):
    """
    Activation layer class
    Consists:
        ReLU
    """

    LayerName = 'Activation'

    def __init__(self, activate_func=None):
        super(Activation, self).__init__()
        self.activation = activate_func

    @staticmethod
    def relu(ipt):
        """
        Activation ReLU
        :param ipt: Tensor. All input elements <= 0 are replaced with 0.
        :return: Activated tensor
        """
        # print ipt
        activation_relu = torch.clamp(ipt, min=0, max=None)
        # print activation_relu
        return activation_relu

    @staticmethod
    def backward_relu(ipt, grad_output):
        grad_output[ipt <= 0] = 0
        return grad_output


class Criterion(ModelNN):
    """
    Criterion classes
    Consists:
        Softmax
    """

    LayerName = 'Criterion'

    def __init__(self, classifier=None):
        super(Criterion, self).__init__()
        self.classifier = classifier

    @staticmethod
    def softmax(ipt):
        """
        Calculate the softmax output of a given vector.
        """
        op_exp = torch.exp(ipt)
        softmax_func = op_exp / torch.sum(op_exp, dim=1, keepdim=True)
        value, index = torch.max(softmax_func, 1)
        # print value, index
        return [value, index.cpu(), softmax_func]

    def linear(self, opt, target):
        pass

    @staticmethod
    def backward_softmax(softmax, target):
        # computes and returns the gradient of the Loss with
        # respect to the input to this layer.
        d_probs = softmax
        d_probs[range(dset.CIFAR10.batch_size), target] -= 1  # Derivation of gradient of loss
        d_probs /= dset.CIFAR10.batch_size
        return d_probs


class Optimize:
    """Schedules learning rate and saves the optimum paramters"""

    def __init__(self, nn_obj):
        self.nn_alias = nn_obj
        self.lr0 = nn_obj.lr

    def constant(self):
        pass

    def time_decay(self, epoch, decay=0):
        self.nn_alias.lr = self.lr0 / (1 + decay * epoch)

    def step_decay(self, epoch, decay_after=5, drop=0.5):
        if decay_after is epoch:
            self.nn_alias.lr *= drop

    def exp_decay(self, decay, epoch):
        self.nn_alias.lr = (self.lr0 * math.exp(-decay * epoch))

    def set_optim_param(self, epoch=-1):
        """
        Saves optimum parameters while training
        :param epoch:
        """
        if self.nn_alias.loss < self.nn_alias.optimum['Loss']:
            self.nn_alias.optimum['Loss'], self.nn_alias.optimum['Epoch'], self.nn_alias.optimum['Learning rate'] = \
                (self.nn_alias.loss, epoch, self.nn_alias.lr)
            self.nn_alias.optimum['Weights'], self.nn_alias.optimum['Biases'] = \
                (self.nn_alias.weights, self.nn_alias.biases)
        if epoch == self.nn_alias.epochs - 1:
            # Set optimum parameters
            self.nn_alias.weights, self.nn_alias.biases = \
                (self.nn_alias.optimum['Weights'], self.nn_alias.optimum['Biases'])
            print("\nOptimum loss in %d epochs is: %f" %
                  (self.nn_alias.epochs, self.nn_alias.optimum['Loss']))

    def clear_gradients(self):
        pass
