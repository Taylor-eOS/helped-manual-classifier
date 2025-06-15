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

EPOCHS = 5
LEARNING_RATE = 0.06
I_VALUE = 1
label_map = {'header': 0, 'body': 1, 'footer': 2, 'quote': 3, 'exclude': 4}

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
    def __init__(self, input_features=20):  # Increased for global features
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
        
    def extract_all_blocks(self):
        all_blocks = []
        for page_num in range(len(self.doc)):
            page_blocks = self.extract_page_geometric_features(page_num)
            for block in page_blocks:
                block['page_num'] = page_num
                block['block_idx_on_page'] = len([b for b in page_blocks if b == block or all_blocks.index(b) < len(all_blocks)])
            all_blocks.extend(page_blocks)
        self.all_blocks = all_blocks
        self.compute_global_stats()
        return all_blocks
    
    def extract_page_geometric_features(self, page_num):
        pass
    
    def compute_global_stats(self):
        if not self.all_blocks:
            return
        font_sizes = [block['font_size'] for block in self.all_blocks]
        widths = [block['width'] for block in self.all_blocks]
        heights = [block['height'] for block in self.all_blocks]
        letter_counts = [block['letter_count'] for block in self.all_blocks]
        positions = [block['position'] for block in self.all_blocks]
        self.global_stats = {
            'font_size_mean': np.mean(font_sizes),
            'font_size_std': np.std(font_sizes),
            'font_size_min': np.min(font_sizes),
            'font_size_max': np.max(font_sizes),
            'width_mean': np.mean(widths),
            'width_std': np.std(widths),
            'height_mean': np.mean(heights),
            'height_std': np.std(heights),
            'letter_count_mean': np.mean(letter_counts),
            'letter_count_std': np.std(letter_counts),
            'position_mean': np.mean(positions),
            'position_std': np.std(positions),
            'total_blocks': len(self.all_blocks),
            'total_pages': len(self.doc)}
    
    def get_global_features(self, block, doc_width=612, doc_height=792, dump=False):
        original_features = [
            block['odd_even'],
            block['x0'] / doc_width,
            block['y0'] / doc_height,
            block['width'] / doc_width,
            block['height'] / doc_height,
            block['position'],
            block['letter_count'] / 100,
            block['font_size'] / 24,
            block['relative_font_size'],
            block['num_lines'] / 10,
            block['punctuation_proportion'],
            block['average_words_per_sentence'] / 10,
            block['starts_with_number'],
            block['capitalization_proportion'],
            block['average_word_commonality'],
            block['squared_entropy']]
        global_features = []
        if self.global_stats:
            font_size_percentile = self.get_percentile(block['font_size'], 
                                                     [b['font_size'] for b in self.all_blocks])
            global_features.extend([
                font_size_percentile,
                (block['font_size'] - self.global_stats['font_size_mean']) / (self.global_stats['font_size_std'] + 1e-6),
                block['page_num'] / self.global_stats['total_pages'],
                self.is_consistent_across_pages(block)])
        else:
            global_features = [0.5, 0.0, 0.5, 0.0]
        all_features = original_features + global_features
        if dump:
            feature_names = ['odd_even', 'x0_norm', 'y0_norm', 'width_norm', 'height_norm', 
                           'position', 'letter_count_norm', 'font_size_norm', 'relative_font_size', 
                           'num_lines_norm', 'punctuation_proportion', 'avg_words_per_sentence_norm', 
                           'starts_with_number', 'capitalization_proportion', 'avg_word_commonality', 
                           'squared_entropy', 'font_size_percentile', 'font_size_zscore', 
                           'page_position', 'cross_page_consistency']
            with open("debug.csv", "a") as f:
                if f.tell() == 0:
                    f.write("Block," + ",".join(feature_names) + "\n")
                f.write(f"{get_next_block_index()}," + ",".join(f"{feature:.5f}" for feature in all_features) + "\n")
        return all_features
    
    def get_percentile(self, value, all_values):
        if not all_values:
            return 0.5
        return sum(1 for v in all_values if v <= value) / len(all_values)
    
    def is_consistent_across_pages(self, block):
        if not self.all_blocks:
            return 0.0
        similar_blocks = []
        for other_block in self.all_blocks:
            if (other_block['page_num'] != block['page_num'] and
                abs(other_block['y0'] - block['y0']) < 50 and
                abs(other_block['font_size'] - block['font_size']) < 2):
                similar_blocks.append(other_block)
        other_pages = set(b['page_num'] for b in similar_blocks)
        total_other_pages = self.global_stats['total_pages'] - 1
        return len(other_pages) / max(total_other_pages, 1)
    
    def add_training_example(self, block_idx, label, doc_width=612, doc_height=792):
        if block_idx >= len(self.all_blocks):
            return
        block = self.all_blocks[block_idx]
        features = self.get_global_features(block, doc_width, doc_height, True)
        self.training_data.append((features, label_map[label]))
    
    def get_training_data(self):
        if not self.training_data:
            return [], []
        features, labels = zip(*self.training_data)
        return list(features), list(labels)
    
    def train_model(self, epochs=EPOCHS, lr=LEARNING_RATE):
        features, labels = self.get_training_data()
        if not features:
            return self.model
        X_train = torch.tensor(features, dtype=torch.float32)
        y_train = torch.tensor(labels, dtype=torch.long)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        return self.model
    
    def predict_all_blocks(self, doc_width=612, doc_height=792):
        if not self.all_blocks:
            return []
        self.model.eval()
        all_features = [self.get_global_features(block, doc_width, doc_height, False) 
                       for block in self.all_blocks]
        X_test = torch.tensor(all_features, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predictions = torch.max(outputs, 1)
        label_names = ['header', 'body', 'footer', 'quote', 'exclude']
        return [label_names[p] for p in predictions.tolist()]
    
    def get_page_predictions(self, page_num):
        page_blocks = [i for i, block in enumerate(self.all_blocks) 
                      if block['page_num'] == page_num]
        all_predictions = self.predict_all_blocks()
        return [all_predictions[i] for i in page_blocks]

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
    global training_data, normalization_buffer
    features = get_features(block, doc_width, doc_height, True)
    training_data.append((features, label_map[label]))
    normalization_buffer.append(features)

def train_model(model, features, labels, epochs=EPOCHS, lr=LEARNING_RATE):
    weights_file = "weights.pth"
    #if os.path.exists(weights_file):
    #    model.load_state_dict(torch.load(weights_file))
    #    print(f"Loaded weights from {weights_file}")
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
    #torch.save(model.state_dict(), weights_file)
    #print(f"Saved weights to {weights_file}")
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

