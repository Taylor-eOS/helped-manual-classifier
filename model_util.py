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

training_data = []
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
        self.block_classifications = {}
        self.current_page = 0
        self.page_buffer = []
        self.current_page_blocks = []
        self.global_idx_counter = 0
        self.mlp_model = BlockClassifier()
        self.processing_lock = threading.Lock()
        self.root = tk.Tk()
        self.root.title("PDF Block Classifier with MLP")
        self.setup_ui()
        self.process_page(self.current_page)
        self.schedule_next_page_processing(self.current_page + 1)
        self.load_page()
        self.root.mainloop()

    def setup_ui(self):
        next_page_button = tk.Button(self.root, text="Next Page", command=self.next_page)
        next_page_button.pack(side=tk.BOTTOM)

    def update_model_and_predictions(self):
        features, labels = get_training_data()
        if features:
            self.mlp_model = train_model(self.mlp_model, features, labels, epochs=1, lr=0.03)
        pred_labels = predict_blocks(self.mlp_model, self.current_page_blocks, self.doc_width, self.doc_height)
        for idx, block in enumerate(self.current_page_blocks):
            if self.block_classifications[block['global_idx']] == '0':
                self.block_classifications[block['global_idx']] = pred_labels[idx]

    def next_page(self):
        self.update_model_and_predictions()
        label_map = {'Header': 0, 'Body': 1, 'Footer': 2, 'Quote': 3, 'Exclude': 4}
        page_training_examples = []
        for block in self.current_page_blocks:
            label_str = self.block_classifications[block['global_idx']]
            features = get_features(block, self.doc_width, self.doc_height)
            page_training_examples.append((features, label_map[label_str]))
            normalization_buffer.append(features)
        if page_training_examples:
            features, labels = zip(*page_training_examples)
            features, labels = list(features), list(labels)
            self.mlp_model = train_model(self.mlp_model, features, labels, epochs=1, lr=0.03)
        training_data.clear()
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.process_page(self.current_page)
            self.load_page()
        else:
            messagebox.showinfo("Info", "You have reached the last page.")

    def process_page(self, page_number):
        page = self.doc.load_page(page_number)
        blocks = page.get_text("dict")["blocks"]
        self.current_page_blocks = []
        for b in blocks:
            if "lines" in b:
                x0 = b["bbox"][0]
                y0 = b["bbox"][1]
                width = b["bbox"][2] - b["bbox"][0]
                height = b["bbox"][3] - b["bbox"][1]
                position = 0
                text_content = " ".join(line["spans"][0]["text"] for line in b["lines"] if line.get("spans"))
                letter_count = len(text_content)
                font_size = b["lines"][0]["spans"][0]["size"] if b["lines"][0].get("spans") else 12
                relative_font_size = font_size / 24
                num_lines = len(b["lines"])
                punctuation_proportion = calculate_punctuation_proportion(text_content)
                average_words_per_sentence = calculate_average_words_per_sentence(text_content)
                starts_with_number = calculate_starts_with_number(text_content)
                capitalization_proportion = calculate_capitalization_proportion(text_content)
                average_word_commonality = get_word_commonality(text_content)
                squared_entropy = calculate_entropy(text_content) ** 2
                block = {
                    "x0": x0,
                    "y0": y0,
                    "width": width,
                    "height": height,
                    "position": position,
                    "letter_count": letter_count,
                    "font_size": font_size,
                    "relative_font_size": relative_font_size,
                    "num_lines": num_lines,
                    "punctuation_proportion": punctuation_proportion,
                    "average_words_per_sentence": average_words_per_sentence,
                    "starts_with_number": starts_with_number,
                    "capitalization_proportion": capitalization_proportion,
                    "average_word_commonality": average_word_commonality,
                    "squared_entropy": squared_entropy
                }
                block["global_idx"] = self.global_idx_counter
                self.global_idx_counter += 1
                self.current_page_blocks.append(block)
                self.block_classifications[block["global_idx"]] = "0"

    def schedule_next_page_processing(self, next_page_number):
        def process_next():
            if next_page_number < self.total_pages:
                self.process_page(next_page_number)
        threading.Timer(0, process_next).start()

    def load_page(self):
        page = self.doc.load_page(self.current_page)
        pix = page.get_pixmap()
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        imgtk = ImageTk.PhotoImage(image=img)
        label = tk.Label(self.root, image=imgtk)
        label.image = imgtk
        label.pack()
        draw_blocks(self.current_page_blocks, label)

if __name__ == "__main__":
    pdf_path = "input.pdf"
    ManualClassifierGUI(pdf_path)

