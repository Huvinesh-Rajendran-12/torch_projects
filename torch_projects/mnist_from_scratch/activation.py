import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    def __init__(self, activation_func):
        super(Activation, self).__init__()
        self.activation_func = activation_func

    def forward(self, input):
        return self.activation_func(input)

    def backward(self, output_gradient):
        # PyTorch handles backpropagation automatically, but for consistency:
        return output_gradient * self.activation_func(input).detach()

class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__(F.relu)

    def backward(self, output_gradient):
        # PyTorch's autograd already knows how to backprop through ReLU, but for clarity:
        return output_gradient * (self.forward(output_gradient) > 0).float()

# Note: PyTorch's ReLU implementation already includes the forward and backward behavior
# you've described. You typically wouldn't need to define a custom backward method for ReLU
# in PyTorch due to its automatic differentiation capabilities.
