"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """
    self.layers = []
    for i in range(len(n_hidde)):
        if i == 0:
            new_layer = nn.Sequential(
                    nn.Linear(n_inputs, n_hidden[i]),
                    nn.ReLU()
                    )
        else:
            new_layers = nn.Sequential(
                    nn.Linear(n_hidden[i - 1], n_hidden[i]),
                    nn.ReLU()
                    )
        self.layers.append(new_layer)

    if not self.layers:
        self.layers.append(nn.Linear(n_inputs, n_classes))
    else:
        self.layers.append(nn.Linear(n_hidden[-1], n_classes))

    self.layers.append(nn.Softmax())
    self.layers = nn.Sequential(*self.layers)


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

    out = self.layers[0](x)
    for layer in self.layers[1:]:
        out = layer(out)
    return out
