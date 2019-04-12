"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data. 
    """
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module. 
        
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
        """
        
        self.params = {'weight': None, 'bias': None}
        self.params['weight'] = np.random.normal(0.0, 0.0001, size=(out_features, in_features))
        self.params['bias'] = np.zeros((out_features, 1))
        self.grads = {'weight': None, 'bias': None}
        self.grads['weight'] = np.zeros((out_features, in_features))
        self.grads['bias'] = np.zeros((out_features, 1))
    def forward(self, x):
        """
        Forward pass.
        
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        out = x @ self.params['weight'].T + self.params['bias'].T
        self.x = x
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        dx = dout @ self.params['weight']
        dw = (self.x.T @ dout).T
        self.grads['weight'] = dw
        self.grads['bias'] = np.expand_dims(np.sum(dout, axis=0), axis=1)
        return dx

    def update_weights(self, learning_rate):
        self.params['weight'] = self.params['weight'] - learning_rate * self.grads['weight']
        self.params['bias'] = self.params['bias'] - learning_rate * self.grads['bias']

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """

    self.x = x
    out = (x>0) * x
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    """

    dx = dout * (self.x>0).astype(float)
    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """
    #print(f"Shape of x: {x.shape}")
    b = x.max(axis=1).reshape(x.shape[0], 1)

    exponent = np.exp(x - b)
    out = exponent / exponent.sum(axis=1).reshape(x.shape[0], 1)
    self.out = out
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    tmp_shape = self.out.shape
    tmp1 = np.zeros((tmp_shape[0], tmp_shape[1], tmp_shape[1]))
    np.einsum('ijj->ij', tmp1)[:] = self.out
    tmp2 = np.einsum('ij, ik->ijk', self.out, self.out)
    tmp3 = tmp1 - tmp2
    dx = np.einsum('ij, ijk->ik', dout,  tmp3)
    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    self.x = x
    self.y = y
    out = - np.sum(y * np.log(x + 1e-9))
    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """
    dx = -y / (x + 1e-9)
    dx /= y.shape[0]
    return dx
