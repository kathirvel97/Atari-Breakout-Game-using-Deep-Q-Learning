#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DuelingDQN, self).__init__()
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        
        # Separate streams for value and advantage
        self.value_fc = nn.Linear(512, 1)
        self.advantage_fc = nn.Linear(512, num_actions)

        # Initialize weights using Xavier uniform
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        
        # Calculate value and advantage streams
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Combine value and advantage into Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
