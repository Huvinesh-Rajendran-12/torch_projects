import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolutional(nn.Module):
    def __init__(self, input_shape, kernel_size, depth):
        super(Convolutional, self).__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)

        # PyTorch's Conv2d layer already includes bias
        self.conv = nn.Conv2d(in_channels=input_depth, out_channels=depth, kernel_size=kernel_size, bias=True)

        # Initialize weights similarly to JAX's random initialization
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None: nn.init.zeros_(self.conv.bias)

    def forward(self, input):
        # PyTorch expects input in shape (N, C, H, W)
        return self.conv(input)

    def backward(self, output_gradient, learning_rate):
        # PyTorch handles backpropagation automatically during training
        # However, if you want to manually adjust weights:
        self.conv.weight.grad = output_gradient
        with torch.no_grad():
            self.conv.weight.sub_(learning_rate * self.conv.weight.grad)
            self.conv.bias.sub_(learning_rate * self.conv.bias.grad)
        return F.conv_transpose2d(output_gradient, self.conv.weight, stride=1, padding=0)



class Reshape(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super(Reshape, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return input.view(self.output_shape)

    def backward(self, output_gradient:torch.Tensor) -> torch.Tensor:
        return output_gradient.view(self.input_shape)
