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
    def __init__(self, input_features=20):
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

class DocumentProcessor:
    def __init__(self, doc):
        self.doc = doc
        self.all_blocks = []
        self.global_stats = {}
        self.training_data = []
        self.model = BlockClassifier()

def compute_norm_params(buffer):
    arr = np.array(buffer)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    return means, stds

def get_features(block, doc_width = 612, doc_height = 792, dump = False): #features_here
    base = []
    for name in settings.BASE_FEATURES:
        value = block[name]
        scale = settings.SCALES.get(name)
        if scale == 'doc_width':
            value /= doc_width
        elif scale == 'doc_height':
            value /= doc_height
        elif isinstance(scale, (int, float)):
            value /= scale
        base.append(value)
    if self.global_stats:
        p = percentile(block['font_size'], self.all_blocks)
        z = (block['font_size'] - self.global_stats['font_size_mean']) / (self.global_stats['font_size_std'] + 1e-6)
        pg = block['page_num'] / self.global_stats['total_pages']
        c = self.is_consistent_across_pages(block)
        base += [p, z, pg, c]
    if dump:
        suffix_names = ['font_size_pct', 'font_size_z', 'page_frac', 'consistency']
        names = settings.BASE_FEATURES + suffix_names
        with open(settings.feature_data_file, 'a') as f:
            if f.tell() == 0:
                f.write("Block," + ",".join(names) + "\n")
            f.write(f"{get_next_block_index()}," + ",".join(f"{v:.5f}" for v in base) + "\n")
    return base

