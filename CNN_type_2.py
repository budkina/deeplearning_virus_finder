import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

import argparse
from gensim.models import Word2Vec
import multiprocessing

class CNN(nn.Module):
    def __init__(self, fragment_length, conv_layers_num, conv_kernel_size, pool_kernel_size, conv_dilation = 1, pool_dilation = 1, conv_stride = 1, pool_stride = 2):
        super(CNN, self).__init__()

        self.input_channels = 4
        self.fragment_length=fragment_length
        self.conv_layers_num = conv_layers_num
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size

        self.conv1_forward = nn.Conv1d(in_channels=self.input_channels,
            out_channels=self.conv_layers_num,
            kernel_size=self.conv_kernel_size,
            stride = conv_stride,
            dilation = conv_dilation)

        self.conv1_reverse = nn.Conv1d(in_channels=self.input_channels,
            out_channels=self.conv_layers_num,
            kernel_size=self.conv_kernel_size,
            stride = conv_stride,
            dilation = conv_dilation)

        self.pool_forward = nn.MaxPool1d(kernel_size=self.pool_kernel_size,
            stride=pool_stride,
            dilation = pool_dilation)

        self.pool_reverse = nn.MaxPool1d(kernel_size=self.pool_kernel_size,
            stride=pool_stride,
            dilation = pool_dilation)


        size_after_conv = (self.fragment_length + 2*0 - conv_dilation*(self.conv_kernel_size-1) - 1) / conv_stride + 1
        size_after_pool = (size_after_conv + 2*0 - pool_dilation*(self.pool_kernel_size-1) - 1) / pool_stride + 1

        self.fc_size = int(size_after_pool)*self.conv_layers_num

        self.fc_forward = nn.Linear(self.fc_size, 2)
        self.fc_reverse = nn.Linear(self.fc_size, 2)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x_forward, x_reverse):
        conv_result_forward=self.conv1_forward(x_forward)
        conv_result_reverse=self.conv1_reverse(x_reverse)
        relu_result_forward = F.relu(conv_result_forward)
        relu_result_reverse = F.relu(conv_result_reverse)
        pooling_result_forward = self.pool_forward(relu_result_forward)      
        pooling_result_reverse = self.pool_reverse(relu_result_reverse)
        fc_result_forward = self.fc_forward(pooling_result_forward.view(-1, self.fc_size))
        fc_result_reverse = self.fc_reverse(pooling_result_reverse.view(-1, self.fc_size))
        relu_result_forward = F.relu(fc_result_forward)
        relu_result_reverse = F.relu(fc_result_reverse)
        result_forward = self.softmax(relu_result_forward)
        result_reverse = self.softmax(relu_result_reverse)
        result = (result_forward + result_reverse)/2
        return result





