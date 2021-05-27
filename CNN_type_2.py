import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import sys
import argparse
import multiprocessing
import logging

class CNN(nn.Module):
    def __init__(self,
        fr_result,
        fragment_length,
        conv_layers_num,
        conv_kernel_size,
        pool_kernel_size,
        fc_size,
        conv_dilation = 1,
        pool_dilation = 1,
        conv_stride = 1,
        pool_stride = 2):
        super(CNN, self).__init__()

        self.input_channels = 4
        self.fr_result = fr_result
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

        self.dropout = nn.Dropout()

        self.input_fc = int(size_after_pool)*self.conv_layers_num
        self.output_fc = fc_size

        self.fc_forward1 = nn.Linear(self.input_fc, self.output_fc)
        self.fc_reverse1 = nn.Linear(self.input_fc, self.output_fc)
        
        self.fc_forward2 = nn.Linear(self.output_fc, 2)
        self.fc_reverse2 = nn.Linear(self.output_fc, 2)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x_forward, x_reverse):
        conv_result_forward=self.conv1_forward(x_forward)
        conv_result_reverse=self.conv1_reverse(x_reverse)
        relu_result_forward = F.relu(conv_result_forward)
        relu_result_reverse = F.relu(conv_result_reverse)
        pooling_result_forward = self.pool_forward(relu_result_forward)
        pooling_result_reverse = self.pool_reverse(relu_result_reverse)


        dropout_result_forward1 = self.dropout(pooling_result_forward)
        dropout_result_reverse1 = self.dropout(pooling_result_reverse)


        fc_result_forward1 = self.fc_forward1(dropout_result_forward1.view(-1, self.input_fc))
        fc_result_reverse1 = self.fc_reverse1(dropout_result_reverse1.view(-1, self.input_fc))

        relu_result_forward1 = F.relu(fc_result_forward1)
        relu_result_reverse1 = F.relu(fc_result_reverse1)


        dropout_result_forward2 = self.dropout(relu_result_forward1)
        dropout_result_reverse2 = self.dropout(relu_result_reverse1)

        fc_result_forward2 = self.fc_forward2(dropout_result_forward2)
        fc_result_reverse2 = self.fc_reverse2(dropout_result_reverse2)

        relu_result_forward2 = F.relu(fc_result_forward2)
        relu_result_reverse2 = F.relu(fc_result_reverse2)

        result_forward = self.softmax(relu_result_forward2)
        result_reverse = self.softmax(relu_result_reverse2)
        
        if self.fr_result=="average":
            result = (result_forward + result_reverse)/2
        elif fr_result =="max":
            result = max(result_forward, result_reverse)
        else:
            logging.error("Unknown fr_result")
            sys.exit()

        return result





