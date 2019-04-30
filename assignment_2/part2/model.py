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

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.seq_length  = seq_length
        self.vocab_size  = vocabulary_size
        self.num_hidden  = lstm_num_hidden
        self.batch_size  = batch_size
        self.device      = device

        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers, batch_first=True).to(device)
        self.output_map = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.softm = nn.Softmax(2)

    def forward(self, x):
        h_n, c_n = self.lstm(x)
        p = self.output_map(h_n)
        y_hat = self.softm(p)
        return y_hat
