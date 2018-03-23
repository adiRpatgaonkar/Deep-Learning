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
from libs.check_args import arguments, using_gpu
from libs.setup import default_tensor_type

from configs.config_model import configs

from data import dataset as dset


def normalize(data, data_size):
    """ Normalizes the given data with 
    mean and standard deviation """
    # flatten
    data = data.view(data_size, -1)
    mean = torch.mean(data, 1, keepdim=True)
    std_deviation = torch.std(data, 1, keepdim=True)
    # print mean, std_deviation  
    data = data - mean
    data = data / std_deviation
    return data


def batch_norm():
    """ TBD """
    pass


class ModelNN(object):
    """ Model class encapsulating torche all layers, 
    functions, hyper parameters etc. """

    def __init__(self):
        # Model status
        self.fitted = self.trained = \
            self.tested = self.infered = False
        # Model type
        self.model_type = ""
        # Net structure
        self.arch, self.layers, self.num_layers, \
        self.train_loss_history, self.val_loss_history, \
        self.crossval_acc_history, self.val_acc_history = "", [], 0, [], [], [], []
        # Parameters
        [self.weights, self.biases, self.output, 
         self.train_loss, self.val_loss] = [], [], [], 0, 0
        self.grad_weights, self.grad_biases, \
        self.grad_output = [], [], []
        # Hyper parameters
        self.lr_policy = ""
        self.weights_decay = self.epochs = \
            self.lr = self.decay_rate = 1
        self.reg = 1e-3  # regularization strength
        # Results
        self.predictions = self.train_acc = self.test_acc = 0
        self.data_set = ""
        self.optimum = {
            'Fitting tested': self.fitted, 'Trained': self.trained, 'Tested': self.tested,
            'Inferenced': self.infered, 'Model type': self.model_type, 'Arch': self.arch,
            'Num layers': self.num_layers, 'Layer objs': self.layers, 'Weights': 0, 'Biases': 0,
            'Max epochs': self.epochs, 'Epoch': 0, 'Learning rate': self.lr,
            'L.R. policy': self.lr_policy, 'Weights decay': self.weights_decay,
            'L.R. decay': self.decay_rate, 'Reg': self.reg, 
            'Training loss': float("inf"), 'Validation loss': float("inf"),
            'TrainAcc': self.train_acc, 'TestAcc': self.test_acc
        }
        # Model running in mode
        self.isTrain = False

    def show_log(self, arch=False, fit=False, train=False, test=False, infer=False, curr_status=None):
        """ Print out stats of the current activity """
        if curr_status:
            print("\nModel status (current):")
            print("{ Fitting tested:", self.optimum['Fitting tested'], "|", "Trained:", self.optimum['Trained'], "|", 
                "Tested:", self.optimum['Tested'], "|", "Inferenced:", self.optimum['Inferenced'], "}")
            print("{Training loss:", self.optimum['Training loss'], "||", "Training accuracy:", self.optimum['TrainAcc'], "% }")
            print("{Validation loss:", self.optimum['Validation loss'], "||", self.optimum['TestAcc'], "% }\n")
        if arch:
            self.show_arch()
        if fit:
            print('FIT {', end="\n ")
        elif train:
            print('TRAIN {', end="\n ")
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
            print('TYPE:', self.model_type, '\n', 'NUM-LAYERS:', self.num_layers, '\n',
                  'EPOCHS:', self.epochs, '\n', 'L.R.:', self.lr, '\n',
                  'LR-POLICY:', self.lr_policy, '\n', 'WEIGHTS-DECAY:', self.weights_decay, '\n',
                  'DECAY-RATE:', self.decay_rate, '\n', 'REG-STRENGTH:', self.reg, '\n',
                  'TRAINING LOSS:', self.optimum['Training loss'], end=" }\n\n")

    def set_logs(self):
        """ Save/update model status, params too
        (constant params; variable params are stored in set_optim_param) """
        [self.optimum['Fitting tested'], self.optimum['Trained'], self.optimum['Tested'], 
        self.optimum['Inferenced']] = self.fitted, self.trained, self.tested, self.infered
        [self.optimum['Model type'], self.optimum['Num layers'], self.optimum['Arch'],
         self.optimum['Layer objs'], self.optimum['Max epochs'], self.optimum['L.R. policy'],
         self.optimum['Weights decay'], self.optimum['L.R. decay'], self.optimum['Reg']] = \
            [self.model_type, self.num_layers, self.arch,
             self.layers, self.epochs, self.lr_policy,
             self.weights_decay, self.decay_rate, self.reg]

    def get_logs(self):
        cfg = configs()
        """ Rebuilding model while loading the status dictionary """
        # NOTE: L.R. has been set to the last L.R. in the latest training round
        self.arch, self.num_layers, self.layers, self.lr, self.decay_rate = (
            self.optimum['Arch'], self.optimum['Num layers'], self.optimum['Layer objs'],
            self.optimum['Learning rate'], self.optimum['L.R. decay'])
        self.model_type = cfg["MODEL"]["TYPE"]
        self.weights_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
        if arguments().FIT:
            mode = "FIT"
        elif arguments().TRAIN:
            mode = "TRAIN"
            # If loaded model has never been trained, 
            # then set L.R. as base lr.
            if not self.optimum['Trained']:
                self.lr = cfg[mode]["BASE_LR"]
                self.optimum['Training loss'] = self.train_loss = \
                self.optimum['Validation loss'] = self.val_loss = \
                float("inf")
        # Constant params
        if arguments().FIT or arguments().TRAIN:
            self.lr_policy = cfg[mode]["LR_POLICY"]
            self.decay_rate = cfg[mode]["DECAY_RATE"]
            self.epochs = cfg[mode]["EPOCHS"]
        # For all model working modes
        [self.weights, self.biases] = (self.optimum['Weights'], 
            self.optimum['Biases'])
        # Set layer weights and biases (for fprop)
        i = 0
        for layer in self.layers:
            if layer.LayerName == 'Linear':
                layer.w = self.optimum['Weights'][i]
                layer.b = self.optimum['Biases'][i]
                i += 1

    def update_parameters(self):
        """ Bias and weight updates """
        for i, (grad_ws, grad_bs) in enumerate(zip(self.grad_weights, self.grad_biases)):
            grad_ws += self.reg * self.weights[i]
            self.weights[i] += (-self.lr * grad_ws)
            self.biases[i] += (-self.lr * grad_bs)

    def parameters(self):
        return [self.weights, self.biases]

    def add(self, layer_obj):
        """ Add layers, activations to the nn architecture """
        self.layers.append(layer_obj)
        if layer_obj.LayerName == 'Linear':
            layer_obj.w *= self.weights_decay
            self.weights.append(layer_obj.w)
            self.biases.append(layer_obj.b)
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
            self.arch += '( ' + str(l.ipt_neurons) + ' x ' + str(l.opt_neurons) + ' )'
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

    def train(self, ipt, label):
        """ Fprop and Backprop to train """
        self.isTrain = True
        ipt = normalize(ipt, ipt.size(0))
        self.forward(ipt, label)
        self.backward(ipt, label)

    def test(self, ipt, target):
        """ Fprop to test torche model """
        self.isTrain = False
        self.forward(normalize(ipt, ipt.size(0)), target)

    def forward(self, ipt, label):
        """ Fprop for sequential NN layers """
        for lth in range(self.num_layers):
            if lth == 0:  # Input layer
                if self.layers[lth].LayerName == 'Linear':
                    self.output[lth] = self.layers[lth].forward(ipt)
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
                self.CELoss(self.output[lth], label)

    def backward(self, ipts, targets):
        """ Backprop for sequential NN layers """
        param = len(self.weights) - 1
        for lth in range(self.num_layers - 1, -1, -1):
            if lth == self.num_layers - 1:  # Last layer
                if self.layers[lth].LayerName == 'Criterion':
                    if self.layers[lth].classifier == 'Softmax':
                        self.grad_output[lth] = \
                            (self.layers[lth].backward_softmax(self.output[lth], targets))
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

    def CELoss(self, softmax, targets):
        """ Cross-entropy loss """
        if self.isTrain:
            correct_log_probs = (-(torch.log(softmax[range(dset.CIFAR10.batch_size), targets])
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
            self.val_loss = torch.sum(probs) / dset.CIFAR10.test_size
        # If fitting/training/testing loss is destroyed.    
        if math.isnan(self.train_loss) or math.isnan(self.val_loss):
            # Quit if loss is NaN.
            print('Loss is NaN\nExiting ...\n')
            if using_gpu():
                torch.cuda.empty_cache()
            sys.exit(1)

    def plot_history(self, loss_history, accuracy_history):
        """ Plot gradient descent curve """
        if loss_history is True:
            output_file("outputs/loss_plots/loss_history.html")
            p = figure(title="Losses", x_axis_label="Num epochs", 
                    y_axis_label="Loss")
            p.line(range(len(self.train_loss_history)), self.train_loss_history, 
                legend='Training loss', line_color='red', line_width=2.1)
            p.line(range(len(self.val_loss_history)), self.val_loss_history, 
                legend='Validation loss', line_color='green', line_width=2.1)
            show(p)
        if accuracy_history is True:
            output_file("outputs/loss_plots/accuracy_history.html")
            p = figure(title="Accuracies", x_axis_label="Num epochs", 
                    y_axis_label="Loss")
            p.line(range(len(self.crossval_acc_history)), self.crossval_acc_history, 
                legend='Cross val accuracy', line_color='red', line_width=2.1)
            p.line(range(len(self.val_acc_history)), self.val_acc_history, 
                legend='Testing accuracy', line_color='green', line_width=2.1)
            show(p)

class LinearLayer(ModelNN):
    """Linear Layer class"""

    LayerName = 'Linear'

    def __init__(self, num_ipt_neurons, num_opt_neurons):
        # print 'Linear layer created'
        # allocate size for the state variables appropriately
        super(LinearLayer, self).__init__()
        self.ipt_neurons = num_ipt_neurons
        self.opt_neurons = num_opt_neurons
        self.w = torch.rand(num_ipt_neurons, num_opt_neurons).type(default_tensor_type())
        self.b = torch.zeros(1, num_opt_neurons).type(default_tensor_type())

    def forward(self, ipt, target=None):
        # Frop the linear layer
        output = torch.mm(ipt, self.w) + self.b
        return output

    def backward(self, ipt, grad_output, i=-1):
        # Backprop the linear layer
        if i == -1:
            grad_w = torch.mm(ipt.t(), grad_output)
            grad_b = torch.sum(grad_output, dim=0, keepdim=True)
            return [grad_w, grad_b]
        grad_output = torch.mm(grad_output, ipt.t())
        return grad_output


class Activation(ModelNN):
    """ReLU Activation layer class"""

    LayerName = 'Activation'

    def __init__(self, activate_func=None):
        # Different activation function
        super(Activation, self).__init__()
        self.activation = activate_func

    @staticmethod
    def relu(ipt):
        # print ipt
        activation_relu = torch.clamp(ipt, min=0)
        # print activation_relu
        return activation_relu

    @staticmethod
    def backward_relu(ipt, grad_output):
        grad_output[ipt <= 0] = 0
        return grad_output


class CeCriterion(ModelNN):
    """Cross-entropy criterion"""

    LayerName = 'Criterion'

    def __init__(self, classifier=None):
        super(CeCriterion, self).__init__()
        self.classifier = classifier

    @staticmethod
    def softmax(opt):
        opt_exp = torch.exp(opt)
        softmax_func = opt_exp / torch.sum(opt_exp, dim=1, keepdim=True)
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
        # Gradient of loss
        d_probs[range(dset.CIFAR10.batch_size), target] -= 1  
        d_probs /= dset.CIFAR10.batch_size
        return d_probs


class Optimize:
    """Schedules learning rate and saves the optimum paramters"""

    def __init__(self, m_object):
        self.m_alias = m_object
        self.lr0 = m_object.lr

    def time_decay(self, epoch, decay=0):
        self.m_alias.lr = self.lr0 / (1 + decay * epoch)

    def step_decay(self, epoch, decay_after=5, drop=0.5):
        if decay_after is epoch:
            self.m_alias.lr *= drop

    def exp_decay(self, decay, epoch):
        self.m_alias.lr = (self.lr0 * math.exp(-decay * epoch))

    def set_optim_param(self, epoch=-1):
        # Check if you've got the best params
        if self.m_alias.test_acc > self.m_alias.optimum['TestAcc']:
            self.m_alias.optimum['Training loss'], self.m_alias.optimum['Epoch'], \
            self.m_alias.optimum['Learning rate'] = \
                (self.m_alias.train_loss, epoch, self.m_alias.lr)
            self.m_alias.optimum['TestAcc'] = self.m_alias.test_acc
            self.m_alias.optimum['Weights'], self.m_alias.optimum['Biases'] = \
                (self.m_alias.weights, self.m_alias.biases)
            self.m_alias.optimum['Validation loss'] = self.m_alias.val_loss
        # Save best params @ last epoch
        if epoch == self.m_alias.epochs - 1:
            # Set optimum parameters
            self.m_alias.weights, self.m_alias.biases = \
                (self.m_alias.optimum['Weights'], self.m_alias.optimum['Biases'])
            # Print least loss
            print("\nOptimum training loss & validation loss in %d epochs is: %f & %f resp." %
                  (self.m_alias.epochs, 
                    self.m_alias.optimum['Training loss'], 
                    self.m_alias.optimum['Validation loss']))

    def clear_gradients(self):
        pass
