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
from utils import calculate_punctuation_proportion, calculate_average_font_size, calculate_num_lines, calculate_average_words_per_sentence, calculate_starts_with_number, calculate_capitalization_proportion, get_word_commonality, calculate_entropy, process_drop_cap
from gui_core import draw_blocks

I_VALUE = 1

def get_next_block_index():
    global I_VALUE
    current_value = I_VALUE
    I_VALUE += 1
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
    def __init__(self):
        super().__init__()
        self.initial_fc = nn.Linear(16, 64)
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
        block['squared_entropy']
    ]
    feature_names = ['odd_even', 'x0', 'y0', 'width', 'height', 'position', 'letter_count', 'font_size', 'relative_font_size', 'num_lines', 'punctuation_proportion', 'average_words_per_sentence', 'starts_with_number', 'capitalization_proportion', 'average_word_commonality', 'squared_entropy']
    if dump:
        with open("debug.csv", "a") as f:
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

def get_training_data():
    if not training_data:
        return [], []
    features, labels = zip(*training_data)
    return list(features), list(labels)

def add_training_example(block, label, doc_width=612, doc_height=792):
    label_map = {'header': 0, 'body': 1, 'footer': 2, 'quote': 3, 'exclude': 4}
    features = get_features(block, doc_width, doc_height, True)
    training_data.append((features, label_map[label]))
    normalization_buffer.append(features)

def train_model(model, features, labels, epochs=1, lr=0.03):
    if not features:
        return model
    X_train = torch.tensor(features, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model

def predict_blocks(model, blocks, doc_width=612, doc_height=792):
    if not blocks:
        return []
    model.eval()
    X_test = torch.tensor([get_features(b, doc_width, doc_height, False) for b in blocks], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
    label_map = ['header', 'body', 'footer', 'quote', 'exclude']
    return [label_map[p] for p in predictions.tolist()]

