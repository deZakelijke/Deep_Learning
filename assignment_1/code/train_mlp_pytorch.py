"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from mlp_pytorch import MLP
from torch import nn, optim
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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

def get_full_dataset_accuracy(dataset, model, n_inputs, batch_size, total_size):
    """
    Caclulates the accuracy over the whole datatset.

    Args:
        dataset: The test dataset over which the accuracy has to be computed
        model: The model for which the accuracy has to be computed
        n_inputs(int): The size of an input vector of model
        batch_size(int): The size of each batch
        total_size: The complete size of the dataset

    Returns:
        accuracy: scalar float, The average accuracy of each batch for the total
            size of the dataset.
    """
    acc = 0
    for i in range(total_size // batch_size):
        test_batch = dataset.next_batch(batch_size)
        reshaped_input = test_batch[0].reshape(test_batch[0].shape[0], n_inputs)
        torch_input = torch.from_numpy(reshaped_input).float()
        targets = torch.from_numpy(test_batch[1]).long()
        predictions = model(torch_input)
        acc += accuracy(predictions, targets)
    acc /= (total_size // batch_size)
    return acc

def train():
    """
    Performs training and evaluation of MLP model. 
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
  
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
  
    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    n_inputs = 3*32*32
    n_classes = 10
    testset_size = 10000
    trainset_size = 50000
    model = MLP(n_inputs, dnn_hidden_units, n_classes, FLAGS.batch_norm)
    if FLAGS.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    else:
        print('No valif optimizer given. Must be: SGD, Adam')
        sys.exit(0)

    loss_function = nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []

    for i in range(FLAGS.max_steps):
        batch = cifar10['train'].next_batch(FLAGS.batch_size)
        reshaped_input = batch[0].reshape(batch[0].shape[0], n_inputs)
        torch_input = torch.from_numpy(reshaped_input).float()
        targets = torch.from_numpy(batch[1]).long()

        model.zero_grad()
        predictions = model(torch_input)
        loss = loss_function(predictions, torch.max(targets, 1)[1])
        loss.backward()
        optimizer.step()

        if not i % FLAGS.eval_freq:
            train_acc = get_full_dataset_accuracy(cifar10['train'], model, n_inputs, 
                                                 FLAGS.batch_size, trainset_size)
            train_accuracies.append(train_acc)

            test_acc = get_full_dataset_accuracy(cifar10['test'], model, n_inputs, 
                                                 FLAGS.batch_size, testset_size)
            test_accuracies.append(test_acc)
            print(f"Epoch: {i}, accuracy: {(test_acc * 100):.1f}%")


    train_acc = get_full_dataset_accuracy(cifar10['train'], model, n_inputs, 
                                         FLAGS.batch_size, trainset_size)
    train_accuracies.append(train_acc)

    test_acc = get_full_dataset_accuracy(cifar10['test'], model, n_inputs, 
                                         FLAGS.batch_size, testset_size)
    test_accuracies.append(test_acc)
    print(f"Epoch: {FLAGS.max_steps}, accuracy: {(test_acc * 100):.1f}%")

    train_plot = plt.plot(train_accuracies, label="Train accuracy")
    test_plot = plt.plot(test_accuracies, label="Test accuracy")
    plt.legend()
    plt.title(f"Accuracies of Pytorch MLP")
    plt.savefig("Accuracy_plot_mlp_pytorch.pdf", bbox_inches="tight")


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
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='Enables batch normalisation')
    parser.add_argument('--optim', type=str, default='SGD',
                        help='The optimizer to use during training (default: SGD)')
    FLAGS, unparsed = parser.parse_known_args()
  
    main()
