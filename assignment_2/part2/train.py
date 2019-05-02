# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def make_one_hot_encoding(batch, vocab_size):
    """
        Encode data of a batch in a one-hot fashion.

    """
    batch = torch.stack(batch)
    size = batch.shape
    batch = batch.view(-1, 1)
    one_hot_batch = torch.zeros(size[0] * size[1], vocab_size).scatter_(1, batch, 1)
    one_hot_batch = one_hot_batch.view(*size, -1)
    return one_hot_batch

def compute_singe_batch_accuracy(predictions, targets):
    """
        Compute the prediction accuracy of a single batch.

    """
    maximums = predictions.max(2)
    correct = (maximums[1] == targets).float()
    accuracy = correct.sum() / (correct.shape[0] * correct.shape[1])
    return accuracy

def generate_text_sample(model, config, vocab_size, device):
    """
        Generate a sample sequence of characters form the model.

    Samples one random letter from the vocabulary and uses that
    as the first input of the LSTM. By using temperature a character
    is sampled, based on the output of the softmax layer of the LSTM.
    The output character of each time step is used as the input 
    for the next timestep.
    """
    seq_length = config.seq_length
    temperature = config.temp

    rand_code = torch.randint(vocab_size - 1, (1, 1)).long()
    input_letter = torch.zeros(1, 1, vocab_size, device=device)
    input_letter[0, 0, rand_code] = 1
    encoded_letters = [rand_code[0, 0]]

    for i in range(seq_length):
        output = model(input_letter, temperature)
        letter = torch.multinomial(output.view(-1), 1)
        encoded_letters.append(letter)
        input_letter = torch.zeros(1, 1, vocab_size, device=device)
        input_letter[0, 0, letter] = 1
    return torch.Tensor(encoded_letters).tolist()

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0, shuffle=True)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, 
                                dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, config.device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.train_steps):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = make_one_hot_encoding(batch_inputs, dataset.vocab_size)
            batch_targets = torch.stack(batch_targets)

            if config.device == 'cuda:0':
                batch_inputs = batch_inputs.cuda()
                batch_targets = batch_targets.cuda()

            # Only for time measurement of step through network
            t1 = time.time()

            model.zero_grad()
            predictions = model(batch_inputs)

            loss = criterion(predictions.view(-1, dataset.vocab_size), batch_targets.view(-1))
            loss.backward()
            optimizer.step()

            accuracy = compute_singe_batch_accuracy(predictions, batch_targets)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
                      f"Train Step {(step + epoch * len(dataset)):04d}/"
                      f"{(len(dataset) * config.train_steps):04d}, "
                      f"Batch size = {config.batch_size}, "
                      f"Examples/Sec = {examples_per_second:.2f}, "
                      f"Accuracy = {accuracy:.2f}, Loss = {loss:.3f}")

            if step % config.sample_every == 0:
                encoded_letters = generate_text_sample(model, config, dataset.vocab_size, device) 
                print(dataset.convert_to_string(encoded_letters))


    print('Done training.')


 ################################################################################
 ################################################################################
def print_config(config):
    for key, value in vars(config).items():
        print(f"{key} : {value}")

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature used for model sampling.')

    config = parser.parse_args()

    print_config(config)
    # Train the model
    train(config)
