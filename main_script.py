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
from embed_semantic import get_raw_embedding
from model_semantic import SemanticHead, LayoutClassifier
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

    def build_models(self):
        device = settings.device
        input_dim = settings.input_feature_length
        expected = len(settings.BASE_FEATURES) + 4 + 5
        if input_dim != expected:
            raise ValueError(f"input_feature_length {input_dim} != expected {expected}")
        if not hasattr(self, 'semantic_head') or self.semantic_head is None:
            self.semantic_head = SemanticHead(input_dim=384, hidden=128, num_classes=5).to(device)
            self.semantic_optimizer = torch.optim.AdamW(self.semantic_head.parameters(), lr=settings.learning_rate, weight_decay=1e-4)
        if not hasattr(self, 'layout_model') or self.layout_model is None:
            self.layout_model = LayoutClassifier(input_dim=input_dim, hidden=64, num_classes=5).to(device)
            self.layout_optimizer = torch.optim.AdamW(self.layout_model.parameters(), lr=settings.learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        if not hasattr(self, 'recent_buffer'):
            self.recent_buffer = []
        if not hasattr(self, 'training_data'):
            self.training_data = []
        if not hasattr(self, 'semantic_class_counts'):
            self.semantic_class_counts = [0, 0, 0, 0, 0]

    def extract_all_blocks(self):
        all_blocks = []
        all_texts = []
        for page_num in range(self.total_pages):
            page_blocks = extract_page_geometric_features(self.doc, page_num)
            all_blocks.extend(page_blocks)
            all_texts.extend([block['text'] for block in page_blocks])
        embeddings = np.zeros((0, 384))
        if all_texts:
            if self.launch_gui:
                texts_length = len(all_texts)
                print(f"Creating {texts_length} embeddings")
                embeddings = get_raw_embedding(all_texts, texts_length)
            else:
                embeddings = np.zeros((len(all_texts), 384))
        for i, block in enumerate(all_blocks):
            raw = embeddings[i] if i < embeddings.shape[0] else np.zeros(384)
            block['raw_embedding'] = raw.tolist()
        for i, block in enumerate(all_blocks):
            block['global_idx'] = i
        self.all_blocks = all_blocks
        self.block_classifications = ['0'] * len(all_blocks)
        self.compute_global_stats()
        self.build_models()
        return all_blocks

    def get_semantic_logits(self, embedding_np):
        device = settings.device
        self.semantic_head.eval()
        with torch.no_grad():
            t = torch.tensor(np.asarray(embedding_np, dtype=np.float32), device=device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            logits = self.semantic_head(t)
            probs = torch.softmax(logits, dim=1)
        logits_np = logits.cpu().numpy()
        probs_np = probs.cpu().numpy()
        if logits_np.shape[0] == 1:
            return logits_np.squeeze(0), probs_np.squeeze(0)
        return logits_np, probs_np

    def get_global_features(self, block, doc_width, doc_height, for_training, semantic_override=None):
        if getattr(settings, 'dump_features', False) and for_training:
            current_page = block.get('page_num', -1)
            if not hasattr(self, '_last_dump_page') or self._last_dump_page != current_page:
                self._dump_counter = 0
                self._last_dump_page = current_page
        orig = []
        orig_names = []
        for name in settings.BASE_FEATURES:
            v = block.get(name, 0.0)
            scale = settings.SCALES.get(name)
            if isinstance(scale, str):
                denom = locals().get(scale, 1.0)
                v = v / denom if denom else 0.0
            elif scale:
                v = v / scale
            orig.append(float(v))
            orig_names.append(name)
        if self.global_stats and 'font_size' in block:
            fs = block.get('font_size', 0.0)
            all_fs = [b.get('font_size', 0.0) for b in self.all_blocks]
            p = self.get_percentile(fs, all_fs)
            z = (fs - self.global_stats['font_size_mean']) / (self.global_stats['font_size_std'] + 1e-6)
            pg = block.get('page_num', 0) / max(1, self.global_stats.get('total_pages', 1))
            c = float(self.is_consistent_across_pages(block))
            glob = [p, z, pg, c]
        else:
            glob = [0.0, 0.0, 0.0, 0.0]
        glob_names = ['font_size_percentile', 'font_size_zscore', 'page_frac', 'consistency']
        if semantic_override is not None:
            semantic_conf = list(semantic_override)
        else:
            emb = block.get('raw_embedding', [0.0] * 384)
            _, probs = self.get_semantic_logits(emb)
            semantic_conf = probs.tolist()
        semantic_conf = semantic_conf[:5]
        if len(semantic_conf) != 5:
            raise ValueError(f"Semantic head returned {len(semantic_conf)} classes, expected 5")
        semantic_names = [f'semantic_{i}' for i in range(len(semantic_conf))]
        features = orig + glob + semantic_conf
        feature_names = orig_names + glob_names + semantic_names
        if len(features) != settings.input_feature_length:
            raise ValueError(f"Feature length mismatch: got {len(features)}, expected {settings.input_feature_length}")
        if getattr(settings, 'dump_features', False) and for_training:
            self.dump_block_features(features, feature_names)
        return features

    def schedule_retrainer(self):
        if not self.training_data and not self.recent_buffer:
            return
        if (self.page_retrain_count < self.page_retrain_limit or
            self.replay_retrain_count < self.replay_retrain_limit or
            self.recent_buffer):
            self.root.after(self.retrain_delay, self.retrain_tick)

    def retrain_tick(self):
        self.build_models()
        if self.recent_buffer:
            batch_data = self.recent_buffer[:self.max_batch]
            self.recent_buffer = self.recent_buffer[self.max_batch:]
            embeddings = []
            layout_features = []
            labels = []
            for emb, feat, lbl in batch_data:
                embeddings.append(emb)
                layout_features.append(feat)
                labels.append(lbl)
                self.training_data.append((emb, feat, lbl))
            if embeddings:
                self.train_semantic_head(embeddings, labels, epochs=settings.epochs, lr=settings.learning_rate)
            if layout_features:
                self.train_model(features=layout_features, labels=labels, epochs=settings.epochs, lr=settings.learning_rate)
            self.page_retrain_count = 0
            self.replay_retrain_count = 0
            if self.launch_gui:
                print(f"Train: fresh batch: {len(batch_data)} examples")
        elif self.replay_retrain_count < self.replay_retrain_limit and len(self.training_data) > 0:
            self.replay_retrain_count += 1
            replay_size = min(self.max_batch, len(self.training_data))
            replay_data = self.training_data[-replay_size:]
            embeddings = []
            layout_features = []
            labels = []
            for emb, feat, lbl in replay_data:
                embeddings.append(emb)
                layout_features.append(feat)
                labels.append(lbl)
            if embeddings:
                self.train_semantic_head(embeddings, labels, epochs=1, lr=settings.learning_rate * 0.5)
            if layout_features:
                self.train_model(features=layout_features, labels=labels, epochs=1, lr=settings.learning_rate * 0.5)
            if self.launch_gui:
                print(f"Train: Replay {self.replay_retrain_count}/{self.replay_retrain_limit}, Loss: replaying {len(replay_data)} examples")
        else:
            if self.launch_gui:
                print("Train: No data or replay limit reached")

    def fetch_training_batch(self, max_items):
        fb, lb = [], []
        if not self.training_data and not self.recent_buffer:
            return fb, lb
        if self.launch_gui:
            print(f"Buffer size: {len(self.recent_buffer)}, Training pool size: {len(self.training_data)}")
        n_recent = min(len(self.recent_buffer), max_items)
        for _ in range(n_recent):
            emb, feat, lbl = self.recent_buffer.pop(0)
            fb.append(feat)
            lb.append(lbl)
            self.training_data.append((emb, feat, lbl))
        remaining = max_items - n_recent
        if remaining > 0 and self.training_data:
            idxs = np.random.choice(len(self.training_data), size=min(remaining, len(self.training_data)), replace=False)
            for i in idxs:
                _, feat, lbl = self.training_data[i]
                fb.append(feat)
                lb.append(lbl)
        return fb, lb

    def train_semantic_head(self, embeddings, labels, epochs=1, lr=None, print_loss=True):
        if not embeddings or not labels:
            return self.semantic_head
        device = settings.device
        X = torch.tensor(np.asarray(embeddings, dtype=np.float32), dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.long, device=device)
        class_counts = np.bincount(labels, minlength=5)
        total_samples = len(labels)
        smooth = 1.0
        class_weights = torch.tensor([(total_samples + 5 * smooth) / (5 * (count + smooth)) for count in class_counts], dtype=torch.float32, device=device)
        if self.launch_gui:
            formatted_weights = ", ".join(f"{w:.2f}" for w in class_weights.cpu().numpy())
            print(f"Semantic class weights: [{formatted_weights}]")
        optimizer = torch.optim.AdamW(self.semantic_head.parameters(), lr=(lr or settings.learning_rate) * 0.8, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        self.semantic_head.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.semantic_head(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.semantic_head.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if print_loss and self.launch_gui:
                print(f"Semantic head train: Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        return self.semantic_head

    def train_model(self, features, labels, print_loss=True, epochs=settings.epochs, lr=settings.learning_rate):
        if not features or not labels:
            return self.layout_model
        if len(features) != len(labels):
            print(f"Warning: Feature count ({len(features)}) doesn't match label count ({len(labels)})")
            return self.layout_model
        device = settings.device
        X_train = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32, device=device)
        y_train = torch.tensor(labels, dtype=torch.long, device=device)
        optimizer = torch.optim.AdamW(self.layout_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        self.layout_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.layout_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.layout_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if print_loss and len(loader) and self.launch_gui:
                print(f"Train: Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        if settings.debug_model_weight_usage:
            with torch.no_grad():
                weights = self.layout_model.initial_fc.weight.cpu().numpy()
                mean_abs_weights = np.mean(np.abs(weights), axis=0)
                for i, w in enumerate(mean_abs_weights):
                    if self.launch_gui: print(f"Feature {i}: {w:.6f}")
        return self.layout_model

    def add_training_example(self, block, label, doc_width=612, doc_height=792):
        self.build_models()
        lab = label_map[label]
        self.semantic_class_counts[lab] += 1
        emb = np.asarray(block.get('raw_embedding', [0.0] * 384), dtype=np.float32)
        features = self.get_global_features(block, doc_width, doc_height, True)
        self.recent_buffer.append((emb.tolist(), features, lab))
        if not hasattr(self, '_scheduler_started'):
            self._scheduler_started = True
            self.schedule_retrainer()

    def get_local_training_data(self):
        if not self.training_data:
            return [], []
        embeddings = []
        features = []
        labels = []
        for emb, feat, lbl in self.training_data:
            embeddings.append(emb)
            features.append(feat)
            labels.append(lbl)
        return features, labels

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
                if any(lbl == 0 for _, lbl in batch):
                    raise ValueError("Attempting to train on label 0 (unlabeled). Check labeling logic.")
                features, labels = zip(*batch)
                if self.launch_gui:
                    print(f"Training on {len(features)} blocks")
                self.train_model(epochs=settings.epochs, lr=settings.learning_rate, features=list(features), labels=list(labels))
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
        device = settings.device
        self.layout_model.eval()
        self.semantic_head.eval()
        page_features = []
        for block in self.current_page_blocks:
            emb = block.get('raw_embedding', [0.0] * 384)
            _, probs = self.get_semantic_logits(emb)
            feat = self.get_global_features(block, 612, 792, False, semantic_override=probs)
            page_features.append(feat)
        X_test = torch.tensor(np.asarray(page_features, dtype=np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = self.layout_model(X_test)
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

    def next_page(self):
        if self.current_page_blocks:
            self.finalize_current_page()
        self.current_page += 1
        if self.current_page >= self.total_pages:
            self.finish_classification()
            return
        self.load_current_page()
        if self.launch_gui and self.current_page_blocks:
            print(f"\nSemantic confidences page {self.current_page + 1}:")
            short_label_names = ['h', 'b', 'f', 'q', 'e']
            embs = [block.get('raw_embedding', [0.0] * 384) for block in self.current_page_blocks]
            if embs:
                emb_array = np.array(embs, dtype=np.float32)
                _, probs_np = self.get_semantic_logits(emb_array)
                if probs_np.ndim == 1:
                    probs_np = probs_np[np.newaxis, :]
                sorted_indices = sorted(
                    range(len(self.current_page_blocks)),
                    key=lambda i: (self.current_page_blocks[i]['y0'], self.current_page_blocks[i]['x0']))
                for idx in sorted_indices:
                    block = self.current_page_blocks[idx]
                    p = probs_np[idx]
                    top_idx = int(np.argmax(p))
                    conf = float(p[top_idx])
                    conf_str = " ".join(f"{short_label_names[j]}:{p[j]:.2f}" for j in range(5))
                    text_snip = block['text'].replace('\n', ' ').strip()
                    if len(text_snip) > 4:
                        text_snip = text_snip[:4].rstrip()
                    text_snip = "\"" + text_snip + "\""
                    print(f"{block['global_idx']:3d}: {text_snip:<6} {conf_str} (top {short_label_names[top_idx]} {conf:.2f})")
        self.page_retrain_count = 0
        self.page_retrain_limit = len(self.current_page_blocks)
        self.replay_retrain_count = 0
        self.schedule_retrainer()
        self.status_var.set(f"Page {self.current_page+1}/{self.total_pages}")

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
        self.status_var.set(f"Page {self.current_page+1}/{self.total_pages}")

    def finish_classification(self):
        torch.save(self.layout_model.state_dict(), 'weights_layout.pt')
        torch.save(self.semantic_head.state_dict(), 'weights_semantic.pt')
        if self.launch_gui: print("Model weights saved")
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

    def set_current_label(self, label):
        self.current_label = label
        self.update_button_highlight()

    def update_button_highlight(self):
        update_button_highlight(self.buttons, self.current_label)

    def load_current_page(self):
        return load_current_page(self)

    def draw_blocks(self):
        return draw_blocks(self)

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
    file_name = input("Enter PDF file basename: ").strip() or "test"
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

