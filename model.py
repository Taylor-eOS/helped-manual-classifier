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

def get_features(block, doc_width=612, doc_height=792, dump=False):
    original_features = [
        block['odd_even'],
        block['x0'],
        block['y0'],
        block['width'],
        block['height'],
        block['position'],
        block['letter_count'],
        block['font_size'],
        block['relative_font_size'],
        block['num_lines'],
        block['punctuation_proportion'],
        block['average_words_per_sentence'],
        block['starts_with_number'],
        block['capitalization_proportion'],
        block['average_word_commonality'],
        block['squared_entropy']]
    feature_names = ['odd_even', 'x0', 'y0', 'width', 'height', 'position', 'letter_count', 'font_size', 'relative_font_size', 'num_lines', 'punctuation_proportion', 'average_words_per_sentence', 'starts_with_number', 'capitalization_proportion', 'average_word_commonality', 'squared_entropy']
    if dump:
        with open(settings.feature_data_file, "a") as f:
            if f.tell() == 0:
                f.write("Block," + ",".join(feature_names) + "\n")
            f.write(f"{get_next_block_index()}," + ",".join(f"{feature:.5f}" for feature in original_features) + "\n")
    raw_features = original_features.copy()
    raw_features[3] /= doc_width
    raw_features[4] /= doc_height
    raw_features[6] /= 100
    raw_features[7] /= 24
    raw_features[9] /= 10
    raw_features[11] /= 10
    if len(normalization_buffer) > 0:
        means, stds = compute_norm_params(normalization_buffer)
        norm_features = [(raw - mean) / (std + epsilon) for raw, mean, std in zip(raw_features, means, stds)]
        return norm_features
    return raw_features

