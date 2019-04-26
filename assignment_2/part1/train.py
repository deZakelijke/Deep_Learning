################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def compute_single_batch_accuracy(predictions, targets):
    """
    Computes the accuracy of a singel batch.

    args:
        predictions(batch_size, n_classes): Output of the softmax of the RNN
        targets(batch_size, n_classes): 

    """
    maximums = predictions.max(1)
    correct = (maximums[1].float() == targets.float()).float()
    accuracy = correct.sum() / correct.shape[0]
    return accuracy


def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, 
                           config.num_hidden, config.num_classes,
                           config.batch_size, device)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim, 
                     config.num_hidden, config.num_classes,
                     config.batch_size, device)
    else:
        raise ValueError("Not a valid network architecture, choose RNN or LSTM")

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    accuracies = []
    tmp_acc = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        if config.device == 'cuda:0':
            batch_inputs = batch_inputs.cuda()
            batch_targets = batch_targets.cuda()

        # Only for time measurement of step through network
        t1 = time.time()

        model.zero_grad()
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        accuracy = compute_single_batch_accuracy(predictions, batch_targets)
        tmp_acc.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            accuracies.append(sum(tmp_acc) / 10)
            tmp_acc = []

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    return accuracies


def print_config(config):
    for key, value in vars(config).items():
        print(f"{key} : {value}")

def run_experiments(config):
    n = config.smoothing

    for i in range(config.input_length_steps):
        print_config(config)
        accuracies = train(config)
        accuracies = [sum(accuracies[i:i + n]) / n for i in range(0, len(accuracies), n)]
        accuracies[-1] *= n
        plt.plot(accuracies, label=f"Sequence length: {config.input_length}")
        config.input_length += 1

    plt.legend()
    plt.title("Accuracy plot for vanilla RNN during training")
    plt.savefig("Accuracy_plot_vanilla_rnn.pdf", bbox_inches="tight")

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--input_length_steps', type=int, default=1, help='Number of times the training is rerun with incremented input length')
    parser.add_argument('--smoothing', type=int, default=5, help="Smoothing factor of accuracy plot")

    config = parser.parse_args()

    run_experiments(config)

