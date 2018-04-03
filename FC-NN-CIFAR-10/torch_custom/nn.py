#!~/anaconda2/envs/bin/python
"""
Custom sequential neural network framework

@author: apatgao
"""

# System imports
from __future__ import print_function

import math
import sys

import torch
from bokeh.plotting import figure, output_file, show

# Custom imports
from data import dataset as dset
from libs.check_args import using_gpu
from libs.setup import default_tensor_type


def normalize(data, data_size):
    """ Normalizes the given data with 
    mean and standard deviation """
    # flatten
    data = data.view(data_size, -1)
    # noinspection PyArgumentList
    mean = torch.mean(data, 1, keepdim=True)
    # noinspection PyArgumentList
    std_deviation = torch.std(data, 1, keepdim=True)
    # print mean, std_deviation  
    data = data - mean
    data = data / std_deviation
    return data


def batch_norm():
    """ TBD """
    pass


class ModelNN(object):
    """ Model class encapsulating the all layers,
    functions, hyper parameters etc. """

    def __init__(self):
        self.data_set = ""
        # Model status
        self.fitted = self.trained = self.tested = self.infered = False
        # Type
        self.type = ""
        # Net structure
        self.arch, self.layers, self.num_layers = "", [], 0
        [self.train_loss_history, self.test_loss_history,
         self.crossval_acc_history, self.test_acc_history] = [], [], [], []
        # Parameters
        self.weights, self.biases, self.output = [], [], []
        self.grad_weights, self.grad_biases, self.grad_output = [], [], []
        self.train_loss = self.test_loss = float("inf")
        # Hyper parameters
        self.lr_policy = ""
        self.weights_decay = self.max_epochs = self.lr = self.lr_decay = 1
        self.curr_epoch = self.start_epoch = 0
        self.reg = 1e-3  # regularization strength
        # Results
        self.predictions = self.train_acc = self.test_acc = 0
        self.optimum = {
            'Fitted': self.fitted, 'Trained': self.trained, 'Tested': self.tested, 'Inferenced': self.infered,
            'Type': self.type, 'Arch': self.arch, 'Num-layers': self.num_layers, 'Layers': self.layers,
            'Weights': 0, 'Biases': 0, 'Weights-decay': self.weights_decay,
            'Max-epochs': self.max_epochs, 'Current-epoch': self.curr_epoch, 'Start-epoch': self.start_epoch,
            'L.R': self.lr, 'L.R-policy': self.lr_policy, 'L.R-decay': self.lr_decay,
            'Reg': self.reg, 'Training-loss': self.train_loss, 'Testing-loss': self.test_loss,
            'Training-accuracy': self.train_acc, 'Testing-accuracy': self.test_acc,
            'Train-loss-history': self.train_loss_history, 'Test-loss-history': self.test_loss_history,
            'CrossVal-accuracy-history': self.crossval_acc_history, 'Test-accuracy-history': self.test_acc_history
        }
        # Model running in mode
        self.isTrain = False

    def state_dict(self, key):
        return self.optimum[key]

    def status(self):
        return [self.fitted, self.trained, self.tested, self.infered]

    def net_architecture(self):
        return [self.type, self.arch, self.layers, self.num_layers]

    def parameters(self):
        return [self.weights, self.biases]

    def gradients(self):
        return [self.grad_weights, self.grad_biases, self.grad_output]

    def hyper_parameters(self):
        return [self.max_epochs, self.curr_epoch, self.start_epoch,
                self.lr, self.lr_policy, self.lr_decay, self.weights_decay, self.reg]

    def losses(self):
        return [self.train_loss, self.test_loss]

    def accuracies(self):
        return [self.train_acc, self.test_acc]

    def history(self):
        return [self.train_loss_history, self.test_loss_history,
                self.crossval_acc_history, self.test_acc_history]

    def save_state(self):
        """ 
        Save/update model status, params too
        (constant params; variable params are stored in set_optim_param) 
        """
        [self.optimum['Fitted'], self.optimum['Trained'], self.optimum['Tested'],
         self.optimum['Inferenced']] = self.status()
        [self.optimum['Type'], self.optimum['Arch'], self.optimum['Layers'], 
        self.optimum['Num-layers']] = self.net_architecture()
        [self.optimum['Max-epochs'], self.optimum['Current-epoch'], self.optimum['Start-epoch'],
         self.optimum['L.R'], self.optimum['L.R-policy'], self.optimum['L.R-decay'],
         self.optimum['Weights-decay'], self.optimum['Reg']] = self.hyper_parameters()
        [self.optimum['Weights'], self.optimum['Biases']] = self.parameters()
        self.optimum['Training-loss'], self.optimum['Testing-loss'] = self.losses()
        [self.optimum['Training-accuracy'], self.optimum['Testing-accuracy']] = self.accuracies()
        [self.optimum['Train-loss-history'], self.optimum['Test-loss-history'],
         self.optimum['CrossVal-accuracy-history'], self.optimum['Test-accuracy-history']] = self.history()

    def get_state(self):
        """ Rebuilding model while loading the status dictionary """
        # NOTE: L.R. has been set to the last L.R. in the latest training round
        self.fitted = self.state_dict('Fitted')
        self.trained = self.state_dict('Trained')
        self.tested = self.state_dict('Tested')
        self.infered = self.state_dict('Inferenced')
        self.type = self.state_dict('Type')
        self.num_layers = self.state_dict('Num-layers')
        self.layers = self.state_dict('Layers')
        self.max_epochs = self.state_dict('Max-epochs')
        self.curr_epoch = self.state_dict('Current-epoch')
        self.start_epoch = self.state_dict('Start-epoch')
        self.lr = self.state_dict('L.R')
        self.lr_policy = self.state_dict('L.R-policy')
        self.lr_decay = self.state_dict('L.R-decay')
        self.weights_decay = self.state_dict('Weights-decay')
        self.train_loss = self.state_dict('Training-loss')
        self.test_loss = self.state_dict('Testing-loss')
        self.train_acc = self.state_dict('Training-accuracy')
        self.test_acc = self.state_dict('Testing-accuracy')
        self.crossval_acc_history = self.state_dict('CrossVal-accuracy-history')
        self.test_acc_history = self.state_dict('Test-accuracy-history')

        [self.weights, self.biases] = (self.optimum['Weights'],
                                       self.optimum['Biases'])
        # Set layer weights and biases (for fprop)
        i = 0
        for layer in self.layers:
            if layer.LayerName == 'Linear':
                layer.weight = self.optimum['Weights'][i]
                layer.bias = self.optimum['Biases'][i]
                i += 1

    def show_log(self, arch=False, fit=False, train=False, test=False, infer=False, curr_status=None):
        """ Print out stats of the current activity """
        if curr_status:
            print(self.status())
        if arch:
            self.show_arch()
        if fit:
            print('FIT {', end="\n")
        elif train:
            print('TRAIN {', end="\n")
        elif test:
            print('TEST', end=" ")
            print('( DATASET: %s )' % self.data_set)
            return
        elif infer:
            print('INFER', end=" ")
            print('( DATASET: %s )' % self.data_set)
            return
        if fit or train:
            print('( DATASET: %s )' % self.data_set, end="\n ")
            print('TYPE:', self.type, '\n', 'NUM-LAYERS:', self.num_layers, '\n',
                  'EPOCHS:', self.max_epochs, '\n', 'L.R.:', self.lr, '\n',
                  'LR-POLICY:', self.lr_policy, '\n', 'WEIGHTS-DECAY:', self.weights_decay, '\n',
                  'DECAY-RATE:', self.lr_decay, '\n', 'REG-STRENGTH:', self.reg, '\n',
                  'Training-loss:', self.optimum['Training-loss'], end=" }\n\n")

    def add(self, layer_obj):
        """ Add layers, activations to the nn architecture """
        self.layers.append(layer_obj)
        if layer_obj.LayerName == 'Linear':
            layer_obj.weight *= self.weights_decay
            self.weights.append(layer_obj.weight)
            self.biases.append(layer_obj.bias)
            self.grad_weights.append(0)
            self.grad_biases.append(0)
        self.output.append(0)
        self.grad_output.append(0)
        self.num_layers += 1
        self.patch_arch(layer_obj)

    def patch_arch(self, l):
        """ Patch architecture string """
        self.arch += str(self.num_layers) + ': ' + l.LayerName
        if l.LayerName == 'Linear':
            self.arch += '( ' + str(l.in_features) + ' x ' + str(l.out_features) + ' )'
        elif l.LayerName == 'Activation':
            self.arch += '( ' + l.activation + ' )'
        elif l.LayerName == 'Criterion':
            self.arch += '( ' + l.classifier + ' )'
        self.arch += '-->\n'
        self.optimum['Arch'] += self.arch

    def show_arch(self):
        """ Print network architecture """
        print('\nNet arch:', end=" ")
        print('{')
        print(self.arch, end="")
        print('\t  }')

    def plot_history(self, loss_history, accuracy_history):
        """ Plot gradient descent curve """
        if loss_history is True:
            output_file("outputs/model_history/loss_history.html")
            p = figure(title="Losses", x_axis_label="Num epochs",
                       y_axis_label="Loss")
            if len(self.train_loss_history) != 0:
                p.line(range(len(self.train_loss_history)), self.train_loss_history,
                       legend='Training-loss', line_color='red', line_width=2.1)
            if len(self.test_loss_history) != 0:
                p.line(range(len(self.test_loss_history)), self.test_loss_history,
                       legend='Testing-loss', line_color='green', line_width=2.1)
            show(p)
        if accuracy_history is True:
            output_file("outputs/model_history/accuracy_history.html")
            p = figure(title="Accuracies", x_axis_label="Num epochs",
                       y_axis_label="Accuracy")
            p.line(range(len(self.crossval_acc_history)), self.crossval_acc_history,
                   legend='Cross val accuracy', line_color='red', line_width=2.1)
            p.line(range(len(self.test_acc_history)), self.test_acc_history,
                   legend='Testing accuracy', line_color='green', line_width=2.1)
            show(p)

    def CELoss(self, softmax, labels):
        """ Cross-entropy loss """
        if self.isTrain:
            correct_log_probs = (-(torch.log(softmax[range(dset.CIFAR10.batch_size), labels])
                                   / torch.log(torch.Tensor([10]).type(default_tensor_type()))))
            # print correct_log_probs
            self.train_loss = torch.sum(correct_log_probs) / dset.CIFAR10.batch_size
            weights = self.parameters()
            reg_loss = 0
            for w in weights[0]:
                reg_loss += 0.5 * self.reg * torch.sum(w * w)
            self.train_loss += reg_loss
        else:
            probs = -((torch.log(softmax) /
                       torch.log(torch.Tensor([10]).type(default_tensor_type()))))
            self.test_loss = torch.sum(probs) / dset.CIFAR10.test_size
        # If fitting/training/Testing-loss is destroyed.    
        if math.isnan(self.train_loss) or math.isnan(self.test_loss):
            # Quit if loss is NaN.
            print('Loss is NaN\nExiting ...\n')
            if using_gpu():
                torch.cuda.empty_cache()
            sys.exit(1)

    def update_parameters(self):
        """ Bias and weight updates """
        for i, (grad_ws, grad_bs) in enumerate(zip(self.grad_weights, self.grad_biases)):
            grad_ws += self.reg * self.weights[i]
            self.weights[i] += (-self.lr * grad_ws)
            self.biases[i] += (-self.lr * grad_bs)

    def train(self, inputs, labels):
        """ Fprop and Backprop to train """
        # SGD
        if self.curr_epoch == self.start_epoch and self.isTrain is False:
            print("\n# Stochastic gradient descent #")
            print("Base learning rate: %.4f\n" % self.lr)
        self.isTrain = True
        inputs = normalize(inputs, inputs.size(0))
        self.forward(inputs, labels)
        self.backward(inputs, labels)

    def test(self, inputs, labels):
        """ Fprop to test the model """
        self.isTrain = False
        self.forward(normalize(inputs, inputs.size(0)), labels)

    def forward(self, inputs, labels):
        """ Fprop for sequential NN layers """
        for lth in range(self.num_layers):
            if lth == 0:  # Input layer
                if self.layers[lth].LayerName == 'Linear':
                    self.output[lth] = self.layers[lth].forward(inputs)
            elif lth < self.num_layers - 1:  # Hidden layers
                if self.layers[lth].LayerName == 'Linear':
                    self.output[lth] = \
                        (self.layers[lth].forward(self.output[lth - 1]))
                elif self.layers[lth].LayerName == 'Activation':
                    if self.layers[lth].activation == 'ReLU':
                        self.output[lth] = \
                            (self.layers[lth].relu(self.output[lth - 1]))
            else:  # Last layer
                if self.isTrain:
                    if self.layers[lth].LayerName == 'Criterion':
                        if self.layers[lth].classifier == 'Softmax':
                            _, _, self.output[lth] = \
                                (self.layers[lth].softmax(self.output[lth - 1]))
                else:
                    if self.layers[lth].LayerName == 'Criterion':
                        self.output[lth], self.predictions, _ = \
                            (self.layers[lth].softmax(self.output[lth - 1]))
                self.CELoss(self.output[lth], labels)

    def backward(self, inputs, labels):
        """ Backprop for sequential NN layers """
        param = len(self.weights) - 1
        for lth in range(self.num_layers - 1, -1, -1):
            if lth == self.num_layers - 1:  # Last layer
                if self.layers[lth].LayerName == 'Criterion':
                    if self.layers[lth].classifier == 'Softmax':
                        self.grad_output[lth] = \
                            (self.layers[lth].backward_softmax(self.output[lth], labels))
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
                    (self.layers[0].backward(inputs, self.grad_output[1]))
        self.update_parameters()



class Conv2D(ModelNN):
    """2D Conv layer class"""

    LayerName = 'Convolution 2D'

    def __init__(self, input_dim, filter_dim, num_kernels, pad=0, stride=1):
        super(Conv2D, self).__init__()
        self.filter_dim = filter_dim
        self.filters, self.padding, self.strides = num_kernels, pad, stride
        self.depth, self.height, self.width = input_dim[0:]
        self.output_dim = [0, 0, 0]
        self.output_dim[0] = self.filters
        self.output_dim[1] = ((self.width - filter_dim + 2 * pad) / stride) + 1
        self.output_dim[2] = ((self.height - filter_dim + 2 * pad) / stride) + 1
        self.feature_map_volume = torch.zeros(0, 0)
        self.kernels = 0.01 * torch.randn(self.filters, self.depth, filter_dim, filter_dim)  # print(self.kernels)
        self.biases = torch.ones(self.filters, 1, 1, 1)  # print(self.biases)
        # print(self.output_dim, self.feature_map_volume.size(), self.kernels.size())

    def convolve(self, images):
        """ Convolution op (Auto-correlation i.e. no flipping of kernels)"""
        if torch.is_tensor(images):
            images = images.numpy()

        images = np.pad(images,
                        mode='constant', constant_values=0
                        pad_width=((0, 0), (0, 0), 
                                    (self.padding, self.padding), 
                                    (self.padding, self.padding))
                        )

        images = torch.from_numpy(images).type(torch.FloatTensor) # TODO: Enable cuda compatibilty
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
                self.feature_map_volume = torch.cat((self.feature_map_volume, o1), dim=0)
        self.feature_map_volume.resize_(self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return self.feature_map_volume


class SpatialPool2D(ModelNN):
    """Max/Mean pooling layer class"""
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
    """Linear Layer class"""

    LayerName = 'Linear'

    def __init__(self, in_features, out_features):
        # print 'Linear layer created'
        # allocate size for the state variables appropriately
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.rand(in_features, out_features).type(default_tensor_type())
        self.bias = torch.zeros(1, out_features).type(default_tensor_type())

    def forward(self, inputs):
        # Fprop the linear layer
        output = torch.mm(inputs, self.weight) + self.bias
        return output

    def backward(self, inputs, grad_outputs, i=-1):
        # Backprop the linear layer
        # Gradient for weights and biases
        if i == -1:
            grad_weight = torch.mm(inputs.t(), grad_outputs)
            grad_bias = torch.sum(grad_outputs, dim=0, keepdim=True)
            return [grad_weight, grad_bias]
        # Gradient for outputs
        grad_outputs = torch.mm(grad_outputs, inputs.t())
        return grad_outputs


class Activation(ModelNN):
    """ReLU Activation layer class"""

    LayerName = 'Activation'

    def __init__(self, activate_func=None):
        # Different activation function
        super(Activation, self).__init__()
        self.activation = activate_func

    @staticmethod
    def relu(inputs):
        # print(inputs)
        activations_relu = torch.clamp(inputs, min=0)
        # print(activations_relu)
        return activations_relu

    @staticmethod
    def backward_relu(inputs, grad_outputs):
        grad_outputs[inputs <= 0] = 0
        return grad_outputs


class CeCriterion(ModelNN):
    """Cross-entropy criterion"""

    LayerName = 'Criterion'

    def __init__(self, classifier=None):
        super(CeCriterion, self).__init__()
        self.classifier = classifier

    @staticmethod
    def softmax(output):
        output_exp = torch.exp(output)
        # noinspection PyArgumentList,PyArgumentList
        softmax_func = output_exp / torch.sum(output_exp, dim=1, keepdim=True)
        # noinspection PyArgumentList
        value, index = torch.max(softmax_func, 1)
        # print value, index
        return [value, index.cpu(), softmax_func]

    def linear(self, opt, target):
        pass

    @staticmethod
    def backward_softmax(softmax, labels):
        # computes and returns the gradient of the Loss with
        # respect to the input to this layer.
        d_probs = softmax
        # Gradient of loss
        d_probs[range(dset.CIFAR10.batch_size), labels] -= 1
        d_probs /= dset.CIFAR10.batch_size
        return d_probs


class Optimize:
    """Schedules L.R and saves the optimum parameters"""

    def __init__(self, m_object):
        self.m_alias = m_object
        self.lr0 = m_object.lr

    def time_decay(self):
        self.m_alias.lr = self.lr0 / (1 + self.m_alias.lr_decay * self.m_alias.curr_epoch)

    def step_decay(self, decay_after=5, drop=0.5):
        if self.m_alias.curr_epoch % decay_after == 0:
            self.m_alias.lr *= drop

    def exp_decay(self):
        self.m_alias.lr = (self.lr0 * math.exp(-self.m_alias.lr_decay * self.m_alias.curr_epoch))

    def set_optim_param(self):
        # Check if you've got the best params via accuracies
        if self.m_alias.test_acc > self.m_alias.optimum['Testing-accuracy']:
            print('Better')
            self.m_alias.save_state()

        # Save best params @ last epoch
        if self.m_alias.curr_epoch == self.m_alias.max_epochs - 1:
            # Set optimum parameters
            self.m_alias.weights, self.m_alias.biases = \
                (self.m_alias.optimum['Weights'], self.m_alias.optimum['Biases'])
            # Print loss which gives best accuracy.
            print("Best accuracy: %.2f%%"
                  "\nOptimum Training-loss & "
                  "Testing-loss in %d epochs is: %f & %f resp." %
                  (self.m_alias.optimum['Testing-accuracy'], self.m_alias.max_epochs,
                   self.m_alias.optimum['Training-loss'],
                   self.m_alias.optimum['Testing-loss']))

    def clear_gradients(self):
        pass
