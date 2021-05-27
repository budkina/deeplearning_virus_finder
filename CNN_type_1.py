import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

class CNN(nn.Module):
    def __init__(self,
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

        self.fragment_length=fragment_length
        self.conv_layers_num = conv_layers_num
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.conv1 = nn.Conv1d(in_channels=self.input_channels,
            out_channels=self.conv_layers_num,
            kernel_size=self.conv_kernel_size,
            stride = conv_stride,
            dilation = conv_dilation)

        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel_size,
            stride=pool_stride,
            dilation = pool_dilation)

        size_after_conv = (self.fragment_length + 2*0 - conv_dilation*(self.conv_kernel_size-1) - 1) / conv_stride + 1
        size_after_pool = (size_after_conv + 2*0 - pool_dilation*(self.pool_kernel_size-1) - 1) / pool_stride + 1

        self.dropout = nn.Dropout()

        self.input_fc = int(size_after_pool)*self.conv_layers_num
        self.output_fc = fc_size
        self.fc1 = nn.Linear(self.input_fc, self.output_fc)
        self.fc2 = nn.Linear(self.output_fc, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        conv_result=self.conv1(x)
        relu_result = F.relu(conv_result)
        pooling_result = self.pool(relu_result)      
        fc_input = pooling_result.view(-1, self.input_fc)
        dropout_result1 = self.dropout(fc_input)
        fc_result1 = self.fc1(dropout_result1)
        relu_result1 = F.relu(fc_result1)
        dropout_result2 = self.dropout(relu_result1)
        fc_result2 = self.fc2(dropout_result2)
        relu_result2 = F.relu(fc_result2)
        result = self.softmax(relu_result2)
        return result