import os, string
import fitz
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from collections import Counter
from wordfreq import word_frequency
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from model import BlockClassifier, training_data
from utils import drop_to_file, extract_page_geometric_features
from gui_core import load_current_page, draw_blocks, update_button_highlight
from feature_utils import FeatureUtils
from embed import get_embedding, apply_document_pca
import settings

letter_labels = {'h':'header','b':'body','f':'footer','q':'quote','e':'exclude'}
label_map = {'header': 0, 'body': 1, 'footer': 2, 'quote': 3, 'exclude': 4}

class ManualClassifierGUI(FeatureUtils):
    def __init__(self, pdf_path=None, launch_gui=True):
        self.launch_gui = launch_gui
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
        self.model = BlockClassifier(input_features=settings.input_feature_length)
        self.training_data = []
        if settings.load_pretraining_weights and not launch_gui:
            self.load_model_weights()
        self.root = tk.Tk()
        self.root.title("PDF Block Classifier with Prediction")
        self.current_label = 'body'
        if launch_gui:
            self.setup_ui()
            self.extract_all_blocks()
            self.load_current_page()
        self.recent_buffer = []
        self.retrain_delay = 10
        self.max_batch = settings.training_examples_per_cycle
        self.training_lock = False
        self.page_retrain_count = 0
        self.page_retrain_limit = 0
        self.replay_retrain_count = 0
        self.replay_retrain_limit = settings.max_replay_rounds
        if not launch_gui:
            return
        self.schedule_retrainer()
        self.root.mainloop()

    def schedule_retrainer(self):
        if not self.training_data and not self.recent_buffer: return
        limit = getattr(self, 'page_retrain_limit', 0)
        if self.page_retrain_count < limit or self.replay_retrain_count < self.replay_retrain_limit or self.recent_buffer: self.root.after(self.retrain_delay, self.retrain_tick)

    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, bg = "white")
        self.canvas.pack(fill = tk.BOTH, expand = True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady = 10, fill = tk.X)
        self.buttons = []
        for idx, (text, label) in enumerate(zip(["Header", "Body", "Footer", "Quote", "Excl."], ['header', 'body', 'footer', 'quote', 'exclude'])):
            btn = tk.Button(self.control_frame, text = text, command = lambda l = label: self.set_current_label(l))
            btn.grid(row = 0, column = idx, padx = 1)
            self.buttons.append(btn)
        self.next_btn = tk.Button(self.control_frame, text = "Next", command = self.next_page, bg = "#4CAF50", fg = "black")
        self.next_btn.grid(row = 0, column = 5, padx = 1)
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable = self.status_var, bg = "white")
        self.status_label.pack(pady = 4, fill = tk.X)
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<n>', lambda event: self.next_page())
        self.root.bind('<space>', lambda event: self.next_page())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.zoom = 2
        self.scale = 1.0
        self.geometry_set = False

    def extract_all_blocks(self):
        all_blocks = []
        all_texts = []
        for page_num in range(self.total_pages):
            page_blocks = extract_page_geometric_features(self.doc, page_num)
            all_blocks.extend(page_blocks)
            all_texts.extend([block['text'] for block in page_blocks])
        if all_texts:
            if self.launch_gui:
                texts_length = len(all_texts)
                print(f"Creating {texts_length} embeddings")
                embeddings = apply_document_pca(get_embedding(all_texts, texts_length), settings.embedding_components)
            else:
                embeddings = np.zeros((len(all_texts), settings.embedding_components))
        else:
            embeddings = np.zeros((0, settings.embedding_components))
        for i, block in enumerate(all_blocks):
            for j in range(settings.embedding_components):
                block[f'embed_{j}'] = embeddings[i][j] if j < embeddings.shape[1] else 0.0
        for i, block in enumerate(all_blocks):
            block['global_idx'] = i
        self.all_blocks = all_blocks
        self.block_classifications = ['0'] * len(all_blocks)
        self.compute_global_stats()
        return all_blocks

    def schedule_retrainer(self):
        if not self.training_data and not self.recent_buffer:
            return
        if (self.page_retrain_count < self.page_retrain_limit or
            self.replay_retrain_count < self.replay_retrain_limit or
            self.recent_buffer):
            self.root.after(self.retrain_delay, self.retrain_tick)

    def retrain_tick(self):
        if self.recent_buffer:
            batch, labels = self.fetch_training_batch(self.max_batch)
            self.model = self.train_model(features=batch, labels=labels, epochs=settings.epochs, lr=settings.learning_rate)
            self.page_retrain_count = 0
            self.replay_retrain_count = 0
            if self.launch_gui: print(f"Train: fresh batch: {len(batch)} examples")
        elif self.replay_retrain_count < self.replay_retrain_limit and len(self.training_data) > 0:
            self.replay_retrain_count += 1
            batch, labels = self.fetch_training_batch(self.max_batch)
            self.model = self.train_model(features=batch, labels=labels, epochs=1, lr=settings.learning_rate * 0.5)
            if self.launch_gui: print(f"Train: Replay {self.replay_retrain_count}/{self.replay_retrain_limit}, Loss: replaying {len(batch)} examples")
        else:
            if self.launch_gui: print("Train: No data or replay limit reached")

    def fetch_training_batch(self, max_items):
        fb, lb = [], []
        if not self.training_data and not self.recent_buffer:
            return fb, lb
        if self.launch_gui: print(f"Buffer size: {len(self.recent_buffer)}, Training pool size: {len(self.training_data)}")
        n_recent = min(len(self.recent_buffer), max_items)
        for _ in range(n_recent):
            f, l = self.recent_buffer.pop(0)
            fb.append(f)
            lb.append(l)
            self.training_data.append((f, l))
        remaining = max_items - n_recent
        if remaining > 0 and self.training_data:
            replay_size = min(remaining, len(self.training_data))
            replay_data = self.training_data[-replay_size:]
            self.training_data = self.training_data[:-replay_size]
            for f, l in replay_data:
                fb.append(f)
                lb.append(l)
                self.training_data.insert(0, (f, l))
        return fb, lb

    def train_model(self, features, labels, print_loss=True, epochs=settings.epochs, lr=settings.learning_rate):
        if not features or not labels:
            return self.model
        if len(features) != len(labels):
            print(f"Warning: Feature count ({len(features)}) doesn't match label count ({len(labels)})")
            return self.model
        X_train = torch.tensor(features, dtype=torch.float32)
        y_train = torch.tensor(labels, dtype=torch.long)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        if settings.debug_input_shape and self.launch_gui:
            print(X_train.shape)
            print(settings.input_feature_length)
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
            if print_loss and len(loader) and self.launch_gui:
                print(f"Train: Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        if settings.debug_model_weight_usage:
            with torch.no_grad():
                weights = self.model.initial_fc.weight.cpu().numpy()
                mean_abs_weights = np.mean(np.abs(weights), axis=0)
                for i, w in enumerate(mean_abs_weights):
                    if self.launch_gui: print(f"Feature {i}: {w:.6f}")
        return self.model

    def add_training_example(self, block, label, doc_width=612, doc_height=792):
        features = self.get_global_features(block, doc_width, doc_height, True)
        lab = label_map[label]
        self.recent_buffer.append((features, lab))
        if not hasattr(self, '_scheduler_started'):
            self._scheduler_started = True
            self.schedule_retrainer()

    def get_local_training_data(self):
        if not self.training_data:
            return [], []
        features, labels = zip(*self.training_data)
        return list(features), list(labels)

    def load_current_page(self):
        page_blocks = [block for block in self.all_blocks if block['page_num'] == self.current_page]
        self.current_page_blocks = page_blocks
        self.global_indices = [block['global_idx'] for block in page_blocks]
        self.page_retrain_count = 0
        self.page_retrain_limit = len(page_blocks)
        self.replay_retrain_count = 0

    def update_model_and_predictions(self):
        if self.training_lock or not self.page_buffer:
            return
        self.training_lock = True
        try:
            batch = []
            for item in self.page_buffer:
                block = self.all_blocks[item['global_idx']]
                feat = self.get_global_features(block, 612, 792, True)
                batch.append((feat, label_map[item['label']]))
            if batch:
                if any(lbl == '0' for _, lbl in batch):
                    raise ValueError("Attempting to train on label '0' (unlabeled). Check labeling logic.")
                features, labels = zip(*batch)
                if self.launch_gui: print(f"Training on {len(features)} blocks")
                self.model = self.train_model(epochs=settings.epochs, lr=settings.learning_rate, features=list(features), labels=list(labels))
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
        labels = ['header', 'body', 'footer', 'quote', 'exclude']
        for block in self.current_page_blocks:
            if block['x0'] <= x <= block['x1'] and block['y0'] <= y <= block['y1']:
                gid = block['global_idx']
                curr = self.block_classifications[gid]
                try:
                    idx = labels.index(curr)
                except ValueError:
                    idx = -1
                next_label = labels[(idx + 1) % len(labels)]
                self.select_block(block, next_label, "click")
                self.draw_blocks()
                self.status_var.set(f"Page {self.current_page+1}/{self.total_pages}")
                break

    def on_canvas_click_old(self, event):
        x = event.x / (self.zoom * self.scale)
        y = event.y / (self.zoom * self.scale)
        for block in self.current_page_blocks:
            if (block['x0'] <= x <= block['x1'] and block['y0'] <= y <= block['y1']):
                global_idx = block['global_idx']
                self.block_classifications[global_idx] = self.current_label
                self.manual_overrides.add(global_idx)
                self.add_training_example(block, self.current_label)
                self.page_buffer.append({
                    'text': block['text'],
                    'label': self.current_label,
                    'y0': block['y0'],
                    'x0': block['x0'],
                    'global_idx': global_idx})
                self.draw_blocks()
                self.status_var.set(f"Page {self.current_page+1}/{self.total_pages}")
        if len(self.page_buffer) >= self.max_batch:
            self.update_model_and_predictions()

    def next_page(self):
        if self.current_page_blocks:
            self.finalize_current_page()
        self.current_page += 1
        if self.current_page >= self.total_pages:
            self.finish_classification()
            return
        self.load_current_page()
        self.page_retrain_count = 0
        self.page_retrain_limit = len(self.current_page_blocks)
        self.replay_retrain_count = 0
        self.schedule_retrainer()

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
        key_to_idx = {'h': 0, 'b': 1, 'f': 2, 'q': 3, 'e': 4}
        if key not in key_to_idx:
            print(f"KeyPress: Ignored key: {key}")
            return
        import gui_core
        result = gui_core.classify_next_block(self, key_to_idx[key])
        if isinstance(result, tuple):
            block, label = result
            self.select_block(block, label, "keyboard")
        elif result == "PAGE_DONE":
            self.next_page()

    def finish_classification(self):
        torch.save(self.model.state_dict(), 'weights.pt')
        print("Model weights saved")
        messagebox.showinfo("Complete", "Classification saved")
        self.doc.close()
        self.root.quit()

    def finalize_current_page(self):
        if not self.current_page_blocks:
            return
        for block in self.current_page_blocks:
            global_idx = block['global_idx']
            current_label = self.block_classifications[global_idx]
            if current_label != '0':
                self.add_training_example(block, current_label)
                if False: print(f"Label: '{current_label}'")
                if False: print(f"{block['text'][:50].replace('\n', '\\n')}")
        sorted_blocks = sorted(self.current_page_blocks, key=lambda b: (b['y0'], b['x0']))
        for block in sorted_blocks:
            lbl = self.block_classifications[block['global_idx']]
            if lbl != '0':
                drop_to_file(block['text'], lbl, self.current_page)
        print(f"Finalized page {self.current_page + 1} with {len([b for b in self.current_page_blocks if self.block_classifications[b['global_idx']] != '0'])} labeled blocks")

    def select_block(self, block, label, source="manual"):
        global_idx = block['global_idx']
        self.block_classifications[global_idx] = label
        if source == "manual":
            self.manual_overrides.add(global_idx)
        if False and self.launch_gui: print(f"{source.upper()}: Selected block as '{label}': {block['text'][:30].replace('\n', '').replace('\r', '')}")

    def apply_ml_predictions(self):
        if not self.current_page_blocks:
            return
        pred_labels = self.predict_current_page()
        ml_predicted_count = 0
        for local_idx, block in enumerate(self.current_page_blocks):
            global_idx = block['global_idx']
            if self.block_classifications[global_idx] == '0':
                predicted_label = pred_labels[local_idx]
                self.block_classifications[global_idx] = predicted_label
                self.select_block(block, predicted_label, "ml_prediction")
                ml_predicted_count += 1
                if self.launch_gui: print(f"ML prediction: '{predicted_label}': {block['text'][:50]}...")
        if ml_predicted_count > 0 and self.launch_gui:
            print(f"Applied {ml_predicted_count} ML predictions")

    def on_close(self):
        self.doc.close()
        self.root.destroy()

    def load_model_weights(self):
        if os.path.exists(settings.pretrained_file):
            self.model.load_state_dict(torch.load(settings.pretrained_file))
            print(f"Loaded pretrained weights")

def main():
    file_name = input("Enter PDF file basename: ").strip()
    pdf_path = f"{file_name}.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return
    open("output.json", "w").close()
    if False: open("ground_truth.json", "w").close()
    open(settings.feature_data_file, "w").close()
    print("Extracting the entire file will take a moment...")
    ManualClassifierGUI(pdf_path)
    print("Classification complete")

if __name__ == "__main__":
    main()

