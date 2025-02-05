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
from utils import drop_to_file, calculate_letter_count, calculate_punctuation_proportion, calculate_average_font_size, calculate_num_lines, calculate_average_words_per_sentence, calculate_starts_with_number, calculate_capitalization_proportion, get_word_commonality, calculate_entropy, process_drop_cap
from gui_core import load_current_page, draw_blocks

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Ensure input is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        residual = x
        out = self.fc1(x)
        out = self.relu(self.norm1(out))
        out = self.fc2(out)
        out = self.norm2(out)
        out += residual  # Skip connection
        return self.relu(out)

class BlockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_fc = nn.Linear(15, 64)
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

training_data = deque(maxlen=1)
normalization_buffer = deque(maxlen=50)
epsilon = 1e-6

def compute_norm_params(buffer):
    arr = np.array(buffer)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    return means, stds

def get_features(block, doc_width=612, doc_height=792):
    raw_features = [
        block['x0'],
        block['y0'],
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
        block['squared_entropy']
    ]
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
    label_map = {'Header': 0, 'Body': 1, 'Footer': 2, 'Quote': 3, 'Exclude': 4}
    features = get_features(block, doc_width, doc_height)
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
    X_test = torch.tensor([get_features(b, doc_width, doc_height) for b in blocks], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
    label_map = ['Header', 'Body', 'Footer', 'Quote', 'Exclude']
    return [label_map[p] for p in predictions.tolist()]

class ManualClassifierGUI:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        first_page = self.doc.load_page(0)
        self.doc_width = first_page.rect.width
        self.doc_height = first_page.rect.height
        self.all_blocks = [None] * self.total_pages
        self.block_classifications = []
        self.current_page = 0
        self.page_buffer = []
        self.current_page_blocks = []
        self.global_indices = []
        self.mlp_model = BlockClassifier()
        self.processing_lock = threading.Lock()
        self.root = tk.Tk()
        self.root.title("PDF Block Classifier with MLP")
        self.setup_ui()
        self.process_page(self.current_page)
        self.schedule_next_page_processing(self.current_page + 1)
        self.load_current_page()
        self.root.mainloop()
    
    def update_model_and_predictions(self):
        features, labels = get_training_data()
        if features:
            self.mlp_model = train_model(self.mlp_model, features, labels, epochs=1, lr=0.03)
        pred_labels = predict_blocks(self.mlp_model, self.current_page_blocks, self.doc_width, self.doc_height)
        for local_idx, block in enumerate(self.current_page_blocks):
            if self.block_classifications[block['global_idx']] == '0':
                self.block_classifications[block['global_idx']] = pred_labels[local_idx]
