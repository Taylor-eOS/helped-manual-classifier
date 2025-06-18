import os
import fitz
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from collections import deque
import settings

i_value = 1

def get_next_block_index():
    global i_value
    current_value = i_value
    i_value += 1
    return current_value

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        residual = x
        out = self.fc1(x)
        out = self.relu(self.norm1(out))
        out = self.fc2(out)
        out = self.norm2(out)
        out += residual
        return self.relu(out)

class BlockClassifier(nn.Module):
    def __init__(self, input_features=settings.input_feature_length):
        super().__init__()
        self.initial_fc = nn.Linear(input_features, 64)
        self.relu = nn.ReLU()
        self.resblock1 = ResidualBlock(64, 128)
        self.resblock2 = ResidualBlock(64, 128)
        self.dropout = nn.Dropout(0.2)
        self.final_fc = nn.Linear(64, 5)

    def forward(self, x):
        x = self.relu(self.initial_fc(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.dropout(x)
        return self.final_fc(x)

training_data = []
normalization_buffer = deque(maxlen=50)
epsilon = 1e-6

