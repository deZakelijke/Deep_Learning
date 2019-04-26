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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        
        self.seq_length  = seq_length
        self.input_dim   = input_dim
        self.num_hidden  = num_hidden
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.device      = device
       
        self.fc_gx = nn.Parameter(torch.randn(num_hidden, input_dim, device=device))
        self.fc_gh = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device))
        self.b_g   = nn.Parameter(torch.randn(num_hidden, device=device))

        self.fc_ix = nn.Parameter(torch.randn(num_hidden, input_dim, device=device))
        self.fc_ih = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device))
        self.b_i   = nn.Parameter(torch.randn(num_hidden, device=device))

        self.fc_fx = nn.Parameter(torch.randn(num_hidden, input_dim, device=device))
        self.fc_fh = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device))
        self.b_f   = nn.Parameter(torch.randn(num_hidden, device=device))

        self.fc_ox = nn.Parameter(torch.randn(num_hidden, input_dim, device=device))
        self.fc_oh = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device))
        self.b_o   = nn.Parameter(torch.randn(num_hidden, device=device))

        self.fc_ph = nn.Parameter(torch.randn(num_classes, num_hidden, device=device))
        self.b_p   = nn.Parameter(torch.randn(num_classes, device=device))

        self.tanh  = nn.Tanh()
        self.sigm  = nn.Sigmoid()
        self.softm = nn.Softmax(1)

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        c = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        for j in range(self.seq_length):
            g = self.tanh(x[:, j].unsqueeze(1) @ self.fc_gx.t() + h @ self.fc_gh.t() + self.b_g)
            i = self.sigm(x[:, j].unsqueeze(1) @ self.fc_ix.t() + h @ self.fc_ih.t() + self.b_g)
            f = self.sigm(x[:, j].unsqueeze(1) @ self.fc_fx.t() + h @ self.fc_fh.t() + self.b_g)
            o = self.sigm(x[:, j].unsqueeze(1) @ self.fc_ox.t() + h @ self.fc_oh.t() + self.b_g)
            c = g * i + c * f
            h = self.tanh(c) * o

        p = h @ self.fc_ph.t() + self.b_p
        y_hat = self.softm(p)
        return y_hat
