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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        
        self.seq_length  = seq_length
        self.input_dim   = input_dim
        self.num_hidden  = num_hidden
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.device      = device

        self.fc_hx = nn.Parameter(torch.randn(num_hidden, input_dim, device=device))
        self.fc_hh = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device))
        self.fc_ph = nn.Parameter(torch.randn(num_classes, num_hidden, device=device))
        self.b_h   = nn.Parameter(torch.randn(num_hidden, device=device))
        self.b_p   = nn.Parameter(torch.randn(num_classes, device=device))
        self.tanh  = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden)

        for i in range(self.seq_length):
            tmp = self.fc_hx @ x[:, i].unsqueeze(0)
            h = self.tanh(tmp + h.t() @ self.fc_hh.t() + self.b_h)

        p = h.t() @ self.fc_ph.t() + self.b_p
        y_hat = self.softmax(p)
        return y_hat
