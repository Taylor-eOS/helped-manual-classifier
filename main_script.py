import os
import fitz
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import string
import numpy as np
from math import log2
from collections import Counter
from wordfreq import word_frequency
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from model_util import BlockClassifier, train_model, predict_blocks, add_training_example, get_training_data, get_features, training_data, normalization_buffer, label_map, EPOCHS, LEARNING_RATE
from utils import drop_to_file, calculate_letter_count, calculate_punctuation_proportion, calculate_average_font_size, calculate_num_lines, calculate_average_words_per_sentence, calculate_starts_with_number, calculate_capitalization_proportion, get_word_commonality, calculate_entropy, process_drop_cap, extract_page_geometric_features
from gui_core import load_current_page, draw_blocks, update_button_highlight

letter_labels = {'h':'header','b':'body','f':'footer','q':'quote','e':'exclude'}

class ManualClassifierGUI:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.all_blocks = []
        self.global_stats = {}
        self.block_classifications = []
        self.model_suggestions = {}
        self.manual_overrides = set()
        self.current_page = 0
        self.page_buffer = []
        self.current_page_blocks = []
        self.global_indices = []
        self.model = BlockClassifier(input_features=20)
        self.training_data = []
        self.load_model_weights()
        self.processing_lock = threading.Lock()
        self.root = tk.Tk()
        self.root.title("PDF Block Classifier with Prediction")
        self.current_label = 'body'
        self.setup_ui()
        self.extract_all_blocks()
        self.training_lock = False
        self.load_current_page()
        self.pending_training_data = []
        self.root.mainloop()
        
    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10, fill=tk.X)
        self.buttons = []
        for idx, (text, label) in enumerate(zip(["Header", "Body", "Footer", "Quote", "Excl."],['header', 'body', 'footer', 'quote', 'exclude'])):
            btn = tk.Button(self.control_frame, text=text,command=lambda l=label: self.set_current_label(l))
            btn.grid(row=0, column=idx, padx=1)
            self.buttons.append(btn)
        self.next_btn = tk.Button(self.control_frame, text="Next", command=self.next_page,bg="#4CAF50", fg="black")
        self.next_btn.grid(row=0, column=5, padx=1)
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white")
        self.status_label.pack(pady=4, fill=tk.X)
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.zoom = 2
        self.scale = 1.0
        self.geometry_set = False

    def extract_all_blocks(self):
        all_blocks = []
        for page_num in range(self.total_pages):
            page_blocks = extract_page_geometric_features(self.doc, page_num)
            starting_global_idx = len(all_blocks)
            for i, block in enumerate(page_blocks):
                block['page_num'] = page_num
                block['global_idx'] = starting_global_idx + i
            all_blocks.extend(page_blocks)
        self.all_blocks = all_blocks
        self.block_classifications = ['0'] * len(all_blocks)
        self.compute_global_stats()
        return all_blocks
    
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
            font_size_percentile = self.get_percentile(block['font_size'], [b['font_size'] for b in self.all_blocks])
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
                # Fixed this line - removed undefined function call
                f.write(f"Block_{len(self.training_data)}," + ",".join(f"{feature:.5f}" for feature in all_features) + "\n")
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
    
    def add_training_example(self, block, label, doc_width=612, doc_height=792):
        features = self.get_global_features(block, doc_width, doc_height, True)
        self.training_data.append((features, label_map[label]))
    
    def get_local_training_data(self):
        if not self.training_data:
            return [], []
        features, labels = zip(*self.training_data)
        return list(features), list(labels)
    
    def train_model(self, epochs=EPOCHS, lr=LEARNING_RATE):
        features, labels = self.get_local_training_data()
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
    
    def load_current_page(self):
        page_blocks = [block for block in self.all_blocks if block['page_num'] == self.current_page]
        self.current_page_blocks = page_blocks
        self.global_indices = [block['global_idx'] for block in page_blocks]
        
    def update_model_and_predictions(self):
        if self.training_lock:
            return
        self.training_lock = True
        try:
            features, labels = self.get_local_training_data()
            if features:
                print(f"Training on {len(features)} blocks")
                self.model = self.train_model(epochs=EPOCHS, lr=LEARNING_RATE)
                self.training_data.clear()
            pred_labels = self.predict_current_page()
            for local_idx, global_idx in enumerate(self.global_indices):
                if self.block_classifications[global_idx] == '0':
                    self.block_classifications[global_idx] = pred_labels[local_idx]
            self.model_suggestions = self.block_classifications.copy()
        finally:
            self.training_lock = False
            
    def predict_current_page(self):
        if not self.current_page_blocks:
            return []
        self.model.eval()
        page_features = [self.get_global_features(block, 612, 792, False) for block in self.current_page_blocks]
        X_test = torch.tensor(page_features, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predictions = torch.max(outputs, 1)
        label_names = ['header', 'body', 'footer', 'quote', 'exclude']
        return [label_names[p] for p in predictions.tolist()]

    def on_canvas_click(self, event):
        x = event.x / (self.zoom * self.scale)
        y = event.y / (self.zoom * self.scale)
        for block in self.current_page_blocks:
            if (block['x0'] <= x <= block['x1'] and block['y0'] <= y <= block['y1']):
                global_idx = block['global_idx']
                self.block_classifications[global_idx] = self.current_label
                self.manual_overrides.add(idx)
                self.add_training_example(block, self.current_label)
                self.page_buffer.append({
                    'text': block['text'],
                    'label': self.current_label,
                    'y0': block['y0'],
                    'x0': block['x0'],
                    'global_idx': global_idx})
                self.draw_blocks()
                self.status_var.set(f"Page {self.current_page+1}/{self.total_pages}")
                break

    def next_page(self):
        sorted_blocks = sorted(self.current_page_blocks, key=lambda b: (b['y0'], b['x0']))
        for block in sorted_blocks:
            label = self.block_classifications[block['global_idx']]
            if label not in ['0']:
                drop_to_file(block['text'], label, self.current_page)
        manual_global_indices = {b['global_idx'] for b in self.page_buffer}
        page_training_data = []
        for block in self.current_page_blocks:
            global_idx = block['global_idx']
            label = self.block_classifications[global_idx]
            if global_idx not in manual_global_indices:
                features = get_features(block, doc_width=612, doc_height=792, dump=False)
                page_training_data.append((features, label_map[label]))
        if page_training_data:
            with threading.Lock():
                global training_data
                training_data.extend(page_training_data)
        self.update_model_and_predictions()
        self.page_buffer = []
        self.current_page += 1
        if self.current_page >= self.total_pages:
            self.finish_classification()
            return
        self.load_current_page()

    def set_current_label(self, label):
        self.current_label = label
        self.update_button_highlight()

    def update_button_highlight(self):
        update_button_highlight(self.buttons, self.current_label)

    def load_current_page(self):
        return load_current_page(self)

    def draw_blocks(self):
        return draw_blocks(self)

    def on_key_press(self, event):
        key = event.keysym.lower()
        key_to_idx = {'h':0,'b':1,'f':2,'q':3,'e':4}
        if key not in key_to_idx:
            return
        import gui_core
        result = gui_core.classify_next_block(self, key_to_idx[key])
        if isinstance(result, tuple):
            block, label = result
            self.add_training_example(block, label)
        if result == "PAGE_DONE":
            sorted_blocks = sorted(self.current_page_blocks, key=lambda b: (b['y0'], b['x0']))
            for block in sorted_blocks:
                lbl = self.block_classifications[block['global_idx']]
                if lbl != '0':
                    drop_to_file(block['text'], lbl, self.current_page)
            self.update_model_and_predictions()
            self.page_buffer = []
            self.current_page += 1
            if self.current_page < self.total_pages:
                self.load_current_page()
            else:
                self.finish_classification()

    def finish_classification(self):
        torch.save(self.model.state_dict(), 'weights.pth')
        print("Model weights saved")
        messagebox.showinfo("Complete", "Classification saved")
        self.doc.close()
        self.root.quit()

    def on_close(self):
        self.doc.close()
        self.root.destroy()

    def load_model_weights(self):
        weights_file = "weights_pretrained.pth"
        if os.path.exists(weights_file):
            self.model.load_state_dict(torch.load(weights_file))
            print(f"Loaded pretrained weights")

def main():
    file_name = input("Enter PDF file basename: ").strip()
    pdf_path = f"{file_name}.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return
    open("output.json", "w").close()
    if False: open("ground_truth.json", "w").close()
    open("debug.csv", "w").close()
    ManualClassifierGUI(pdf_path)
    print("Classification complete")

if __name__ == "__main__":
    main()

