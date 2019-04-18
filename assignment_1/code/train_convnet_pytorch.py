"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from convnet_pytorch import ConvNet
from torch import nn, optim
import cifar10_utils

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
  
    maximums = (predictions == predictions.max(1)[0].reshape(predictions.shape[0], 1)).float()
    correct = maximums * targets.float()
    accuracy = correct.sum() / correct.shape[0]

    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model. 
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
  
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
  
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    n_channels = 3
    n_classes = 10
    image_size = (32, 32)
    model = ConvNet(n_channels, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []

    for i in range(FLAGS.max_steps):
        batch = cifar10['train'].next_batch(FLAGS.batch_size)
        reshaped_input = batch[0].reshape(batch[0].shape[0], n_channels, *image_size)
        torch_input = torch.from_numpy(reshaped_input).float()

        targets = torch.from_numpy(batch[1]).long()

        model.zero_grad()
        predictions = model(torch_input)
        loss = loss_function(predictions, torch.max(targets, 1)[1])
        loss.backward()
        optimizer.step()

        if not i % FLAGS.eval_freq:
            train_acc = accuracy(predictions, targets)
            train_accuracies.append(train_acc)

            test_batch = cifar10['test'].next_batch(FLAGS.batch_size * 5)
            torch_input = torch.from_numpy(test_batch[0]).float()
            targets = torch.from_numpy(test_batch[1]).long()
            
            predictions = model(torch_input)

            test_acc = accuracy(predictions, targets)
            test_accuracies.append(test_acc)

            print(f"Epoch: {i}, accuracy: {(test_acc * 100):.1f}%")

    train_acc = accuracy(predictions, targets)
    train_accuracies.append(train_acc)

    test_batch = cifar10['test'].next_batch(FLAGS.batch_size)
    torch_input = torch.from_numpy(batch[0]).float()
    targets = torch.from_numpy(test_batch[1]).long()

    predictions = model(torch_input)
    test_acc = accuracy(predictions, targets)
    test_accuracies.append(test_acc)
    print(f"Epoch: {FLAGS.max_steps}, accuracy: {(test_acc * 100):.1f}%")

    train_plot = plt.plot(train_accuracies, label="Train accuracy")
    test_plot = plt.plot(test_accuracies, label="Test accuracy")
    plt.legend()
    plt.title(f"Accuracies of Pytorch ConvNet")
    plt.savefig("Accuracy_plot_ConvNet_pytorch.pdf", bbox_inches="tight")



def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
  
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
  
    # Run the training operation
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
  
    main()
