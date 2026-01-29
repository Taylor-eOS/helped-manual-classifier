import json
import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from embed_semantic import get_raw_embedding
from model_semantic import SemanticHead, LayoutClassifier
from global_features import build_orig_features, build_global_stat_features, build_semantic_features
from feature_utils import FeatureUtils
import settings

label_map = {'header': 0, 'body': 1, 'footer': 2, 'quote': 3, 'exclude': 4}
reverse_label_map = {0: 'header', 1: 'body', 2: 'footer', 3: 'quote', 4: 'exclude'}

class JSONLClassifierGUI(FeatureUtils):
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.all_blocks = []
        self.global_stats = {}
        self.block_classifications = []
        self.manual_overrides = set()
        self.current_idx = 0
        self.training_data = []
        self.recent_buffer = []
        self.page_buffer = []
        self.root = tk.Tk()
        self.root.title("JSON Lines Block Classifier")
        self.current_label = 'body'
        self.setup_ui()
        self.load_all_blocks()
        self.build_models()
        self.show_current_block()
        self.schedule_retrainer()
        self.root.mainloop()

    def setup_ui(self):
        self.text_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20, font=("Courier", 11))
        self.text_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10, fill=tk.X)
        self.buttons = []
        button_configs = [("Header", 'header'), ("Body", 'body'), ("Footer", 'footer'), ("Quote", 'quote'), ("Excl.", 'exclude')]
        for idx, (text, label) in enumerate(button_configs):
            btn = tk.Button(self.control_frame, text=text, command=lambda l=label: self.set_label_and_next(l))
            btn.grid(row=0, column=idx, padx=5)
            self.buttons.append(btn)
        self.next_btn = tk.Button(self.control_frame, text="Skip", command=self.next_block, bg="#FFA500", fg="black")
        self.next_btn.grid(row=0, column=5, padx=5)
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white")
        self.status_label.pack(pady=4, fill=tk.X)
        key_mappings = {'h': 'header', 'b': 'body', 'f': 'footer', 'q': 'quote', 'e': 'exclude'}
        for key, label in key_mappings.items():
            self.root.bind(key, lambda event, l=label: self.set_label_and_next(l))
        self.root.bind('<space>', lambda event: self.next_block())
        self.root.bind('<n>', lambda event: self.next_block())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_button_highlight()

    def load_all_blocks(self):
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        blocks = []
        page_groups = defaultdict(list)
        for line in lines:
            entry = json.loads(line.strip())
            text = entry.get('text', '')
            page = entry.get('page', 1)
            blocks.append({'text': text, 'page_num': page - 1})
            page_groups[page - 1].append(len(blocks) - 1)
        for page_num, indices in page_groups.items():
            page_height = 792
            num_blocks = len(indices)
            block_height = page_height / max(num_blocks, 1)
            for local_idx, global_idx in enumerate(indices):
                y0 = local_idx * block_height
                y1 = (local_idx + 1) * block_height
                blocks[global_idx]['x0'] = 0
                blocks[global_idx]['y0'] = y0
                blocks[global_idx]['x1'] = 612
                blocks[global_idx]['y1'] = y1
                blocks[global_idx]['width'] = 612
                blocks[global_idx]['height'] = block_height
        all_texts = [block['text'] for block in blocks]
        embeddings = get_raw_embedding(all_texts, len(all_texts))
        for i, block in enumerate(blocks):
            raw = embeddings[i] if i < embeddings.shape[0] else np.zeros(384)
            block['raw_embedding'] = raw.tolist()
            block['global_idx'] = i
        self.all_blocks = blocks
        self.block_classifications = ['0'] * len(blocks)
        self.compute_global_stats()

    def build_models(self):
        device = settings.device
        input_dim = settings.input_feature_length
        self.semantic_head = SemanticHead(input_dim=384, hidden=128, num_classes=5).to(device)
        self.semantic_optimizer = torch.optim.AdamW(self.semantic_head.parameters(), lr=settings.learning_rate, weight_decay=1e-4)
        self.layout_model = LayoutClassifier(input_dim=input_dim, hidden=64, num_classes=5).to(device)
        self.layout_optimizer = torch.optim.AdamW(self.layout_model.parameters(), lr=settings.learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.semantic_class_counts = [0, 0, 0, 0, 0]
        self.retrain_delay = 10
        self.max_batch = settings.training_examples_per_cycle
        self.training_lock = False

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
        orig, orig_names = build_orig_features(block, doc_width, doc_height, settings.BASE_FEATURES, settings.SCALES)
        glob, glob_names = build_global_stat_features(block, self.all_blocks, self.global_stats, self.get_percentile, self.is_consistent_across_pages)
        semantic_conf, semantic_names = build_semantic_features(block, semantic_override, self.get_semantic_logits)
        features = orig + glob + semantic_conf
        if len(features) != settings.input_feature_length:
            raise ValueError(f"Feature length mismatch: got {len(features)}, expected {settings.input_feature_length}")
        return features

    def show_current_block(self):
        if self.current_idx >= len(self.all_blocks):
            self.finish_classification()
            return
        block = self.all_blocks[self.current_idx]
        self.text_display.delete(1.0, tk.END)
        self.text_display.tag_configure("header", font=("Courier", 11, "bold"), foreground="#ff0000")
        self.text_display.tag_configure("body", font=("Courier", 11), foreground="#0066cc")
        self.text_display.tag_configure("footer", font=("Courier", 11), foreground="#2e8b57")
        self.text_display.tag_configure("quote", font=("Courier", 11, "italic"), foreground="#daa520")
        self.text_display.tag_configure("exclude", font=("Courier", 11), foreground="#a0a0a0")
        self.text_display.tag_configure("unlabeled", font=("Courier", 11), foreground="#000000")
        current_label = self.block_classifications[self.current_idx]
        tag = current_label if current_label != '0' else "unlabeled"
        header_text = f"Block {self.current_idx + 1} / {len(self.all_blocks)}\n"
        header_text += f"Page: {block['page_num'] + 1}\n"
        if current_label != '0':
            header_text += f"Current label: {current_label}\n"
        header_text += "=" * 80 + "\n\n"
        self.text_display.insert(tk.END, header_text)
        self.text_display.insert(tk.END, block['text'], tag)
        self.text_display.insert(tk.END, "\n\n" + "=" * 80 + "\n")
        emb = block.get('raw_embedding', [0.0] * 384)
        _, probs = self.get_semantic_logits(emb)
        pred_text = "ML Prediction: "
        label_names = ['header', 'body', 'footer', 'quote', 'exclude']
        for i, name in enumerate(label_names):
            pred_text += f"{name}:{probs[i]:.2f} "
        self.text_display.insert(tk.END, pred_text)
        self.status_var.set(f"Block {self.current_idx + 1}/{len(self.all_blocks)} | Page {block['page_num'] + 1}")

    def set_label_and_next(self, label):
        block = self.all_blocks[self.current_idx]
        global_idx = block['global_idx']
        self.block_classifications[global_idx] = label
        self.manual_overrides.add(global_idx)
        self.add_training_example(block, label)
        self.page_buffer.append({'text': block['text'], 'label': label, 'y0': block['y0'], 'x0': block['x0'], 'global_idx': global_idx})
        self.save_block_to_output(block, label)
        self.next_block()

    def next_block(self):
        self.current_idx += 1
        if self.current_idx >= len(self.all_blocks):
            self.finish_classification()
            return
        self.show_current_block()

    def add_training_example(self, block, label, doc_width=612, doc_height=792):
        lab = label_map[label]
        self.semantic_class_counts[lab] += 1
        emb = np.asarray(block.get('raw_embedding', [0.0] * 384), dtype=np.float32)
        features = self.get_global_features(block, doc_width, doc_height, True)
        self.recent_buffer.append((emb.tolist(), features, lab))
        if not hasattr(self, '_scheduler_started'):
            self._scheduler_started = True

    def schedule_retrainer(self):
        if not self.training_data and not self.recent_buffer:
            self.root.after(self.retrain_delay, self.schedule_retrainer)
            return
        self.root.after(self.retrain_delay, self.retrain_tick)

    def retrain_tick(self):
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
        self.schedule_retrainer()

    def train_semantic_head(self, embeddings, labels, epochs=1, lr=None, print_loss=False):
        if not embeddings or not labels:
            return self.semantic_head
        device = settings.device
        X = torch.tensor(np.asarray(embeddings, dtype=np.float32), dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.long, device=device)
        class_counts = np.bincount(labels, minlength=5)
        total_samples = len(labels)
        smooth = 1.0
        class_weights = torch.tensor([(total_samples + 5 * smooth) / (5 * (count + smooth)) for count in class_counts], dtype=torch.float32, device=device)
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
        return self.semantic_head

    def train_model(self, features, labels, print_loss=False, epochs=None, lr=None):
        if not features or not labels:
            return self.layout_model
        if len(features) != len(labels):
            return self.layout_model
        device = settings.device
        X_train = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32, device=device)
        y_train = torch.tensor(labels, dtype=torch.long, device=device)
        optimizer = torch.optim.AdamW(self.layout_model.parameters(), lr=lr or settings.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        self.layout_model.train()
        for epoch in range(epochs or settings.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.layout_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.layout_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
        return self.layout_model

    def save_block_to_output(self, block, label):
        label_mapping = {"header": "h1", "body": "p", "footer": "footer", "quote": "blockquote", "exclude": "exclude"}
        entry = {"label": label_mapping.get(label, "p"), "page": block['page_num'] + 1, "text": block['text']}
        with open("output.json", "a", encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def finish_classification(self):
        for block in self.all_blocks:
            label = self.block_classifications[block['global_idx']]
            if label == '0':
                continue
            if block['global_idx'] not in self.manual_overrides:
                self.save_block_to_output(block, label)
        torch.save(self.layout_model.state_dict(), 'weights_layout_jsonl.pt')
        torch.save(self.semantic_head.state_dict(), 'weights_semantic_jsonl.pt')
        messagebox.showinfo("Complete", "Classification saved to output.json")
        self.root.quit()

    def set_current_label(self, label):
        self.current_label = label
        self.update_button_highlight()

    def update_button_highlight(self):
        for btn in self.buttons:
            text = btn['text']
            label = 'exclude' if text == 'Excl.' else text.lower()
            btn.config(relief=tk.SUNKEN if label == self.current_label else tk.RAISED)

    def on_close(self):
        for block in self.all_blocks[self.current_idx:]:
            label = self.block_classifications[block['global_idx']]
            if label != '0' and block['global_idx'] in self.manual_overrides:
                self.save_block_to_output(block, label)
        torch.save(self.layout_model.state_dict(), 'weights_layout_jsonl.pt')
        torch.save(self.semantic_head.state_dict(), 'weights_semantic_jsonl.pt')
        self.root.destroy()

def main():
    import sys
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        jsonl_path = input("Enter JSON lines file path: ").strip()
    if not jsonl_path:
        print("Error: No file path provided")
        return
    import os
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found")
        return
    open("output.json", "w").close()
    JSONLClassifierGUI(jsonl_path)

if __name__ == "__main__":
    main()
