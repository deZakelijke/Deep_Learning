"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
    super(ConvNet, self).__init__()
    modules = []

    modules.append(nn.Sequential(
        nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1)
    ))

    modules.append(nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1)
    ))

    modules.append(nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1)
    ))

    modules.append(nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1)
    ))

    modules.append(nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.AvgPool2d(1, stride=1, padding=0),
    ))

    modules.append(nn.Sequential(
        Flatten(),
        nn.Linear(512, n_classes),
        nn.Softmax(dim=1)
    ))

    self.layers = nn.Sequential(*modules)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    out = self.layers(x)

    return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
