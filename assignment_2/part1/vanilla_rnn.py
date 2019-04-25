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

    def forward(self, x):
        print(x.shape)
        print(x[0])
        h = torch.zeros(self.num_hidden)

        for i in range(self.seq_length):
            h = 1 

