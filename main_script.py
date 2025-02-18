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
from model_util import BlockClassifier, train_model, predict_blocks, add_training_example, get_training_data, get_features, training_data, normalization_buffer, label_map
from utils import drop_to_file, calculate_letter_count, calculate_punctuation_proportion, calculate_average_font_size, calculate_num_lines, calculate_average_words_per_sentence, calculate_starts_with_number, calculate_capitalization_proportion, get_word_commonality, calculate_entropy, process_drop_cap, extract_page_geometric_features
from gui_core import load_current_page, draw_blocks, update_button_highlight
from embed import get_embedding

save_weights = False #Turn this on in pretraining
letter_labels = {'h':'header','b':'body','f':'footer','q':'quote','e':'exclude'}

class ManualClassifierGUI:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.all_blocks = [None] * self.total_pages
        self.block_classifications = []
        self.current_page = 0
        self.page_buffer = []
        self.current_page_blocks = []
        self.global_indices = []
        self.mlp_model = BlockClassifier()
        self.processing_lock = threading.Lock()
        self.root = tk.Tk()
        self.root.title("PDF Block Classifier with MLP Help")
        self.current_label = 'body'
        self.setup_ui()
        self.process_page(self.current_page)
        self.schedule_next_page_processing(self.current_page + 1)
        self.training_lock = False
        self.load_current_page()
        self.root.mainloop()
        self.load_model_weights()
        self.pending_training_data = []

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
        self.next_btn = tk.Button(self.control_frame, text="Next", command=self.next_page,bg="#4CAF50", fg="white")
        self.next_btn.grid(row=0, column=5, padx=1)
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white")
        self.status_label.pack(pady=4, fill=tk.X)
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.zoom = 2
        self.scale = 1.0
        self.geometry_set = False

    def process_page(self, page_num):
        if page_num >= self.total_pages:
            return
        with self.processing_lock:
            if self.all_blocks[page_num] is not None:
                return
        page_blocks = extract_page_geometric_features(self.doc, page_num)
        with self.processing_lock:
            if self.all_blocks[page_num] is not None:
                return
            starting_global_idx = sum(len(p) for p in self.all_blocks[:page_num] if p is not None)
            for i, block in enumerate(page_blocks):
                block['global_idx'] = starting_global_idx + i
            self.all_blocks[page_num] = page_blocks
            self.block_classifications.extend(['0'] * len(page_blocks))

    def schedule_next_page_processing(self, next_page_num):
        if next_page_num >= self.total_pages:
            return
        with self.processing_lock:
            if self.all_blocks[next_page_num] is not None:
                return
        threading.Thread(target=self.process_page, args=(next_page_num,)).start()

    def update_model_and_predictions(self):
        if self.training_lock:
            return
        self.training_lock = True
        try:
            features, labels = get_training_data()
            if features:
                print(f"Training on {len(features)} blocks")
                self.mlp_model = train_model(self.mlp_model, features, labels, epochs=5, lr=0.05)
                training_data.clear()
                normalization_buffer.clear()
            pred_labels = predict_blocks(self.mlp_model, self.current_page_blocks)
            for local_idx, global_idx in enumerate(self.global_indices):
                if self.block_classifications[global_idx] == '0':
                    self.block_classifications[global_idx] = pred_labels[local_idx]
        finally:
            self.training_lock = False

    def on_canvas_click(self, event):
        x = event.x / (self.zoom * self.scale)
        y = event.y / (self.zoom * self.scale)
        for block in self.current_page_blocks:
            if (block['x0'] <= x <= block['x1'] and block['y0'] <= y <= block['y1']):
                global_idx = block['global_idx']
                self.block_classifications[global_idx] = self.current_label
                add_training_example(block, self.current_label)
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
        #print(f"Current page: {self.current_page}")
        self.update_model_and_predictions()
        self.page_buffer = []
        self.current_page += 1
        if self.current_page >= self.total_pages:
            self.finish_classification()
            return
        with self.processing_lock:
            if self.all_blocks[self.current_page] is None:
                self.process_page(self.current_page)
        self.schedule_next_page_processing(self.current_page + 1)
        self.load_current_page()

    def set_current_label(self, label):
        self.current_label = label
        self.update_button_highlight()

    def update_button_highlight(self):
        update_button_highlight(self.buttons, self.current_label)

    def extract_page_geometric_features(self, page_num):
        return extract_page_geometric_features(self.doc, page_num)

    def load_current_page(self):
        return load_current_page(self)

    def draw_blocks(self):
        return draw_blocks(self)

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in letter_labels:
            self.set_current_label(letter_labels[key])

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in letter_labels:
            self.set_current_label(letter_labels[key])

    def finish_classification(self):
        if save_weights:
            torch.save(self.mlp_model.state_dict(), 'model_weights.pth')
            print("Model weights saved")
        messagebox.showinfo("Complete", "Classification saved to output file")
        self.doc.close()
        self.root.quit()

    def on_close(self):
        self.doc.close()
        self.root.destroy()

    def load_model_weights(self):
        weights_file = "pretrained_weights.pth"
        if os.path.exists(weights_file):
            self.mlp_model.load_state_dict(torch.load(weights_file))
            print(f"Loaded pre-trained weights from {weights_file}")

def main():
    file_name = input("Enter PDF filename (without extension): ").strip()
    pdf_path = f"{file_name}.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return
    open("output.json", "w").close()
    open("ground_truth.json", "w").close()
    open("debug.csv", "w").close()
    ManualClassifierGUI(pdf_path)
    print("Classification complete")

if __name__ == "__main__":
    main()

