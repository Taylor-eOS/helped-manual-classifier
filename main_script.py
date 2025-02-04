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
from mlp_model import BlockClassifier, train_model, predict_blocks, add_training_example, get_training_data
from utils import calculate_height, calculate_width, calculate_position, calculate_letter_count, calculate_punctuation_proportion, calculate_average_font_size, calculate_num_lines, calculate_average_words_per_sentence, calculate_starts_with_number, calculate_capitalization_proportion, get_word_commonality, calculate_entropy, process_drop_cap

class ManualClassifierGUI:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.all_blocks = self.extract_geometric_features()
        self.current_page = 0
        self.block_classifications = ['0'] * len(self.all_blocks)
        self.current_label = 'Body'
        self.page_buffer = []
        self.current_page_blocks = []
        self.global_indices = []
        
        # Initialize MLP model
        self.mlp_model = BlockClassifier()  # <-- THIS FIXES THE ERROR
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("PDF Block Classifier with MLP")
        self.setup_ui()
        self.load_current_page()
        self.root.mainloop()

    def setup_ui(self):
        # Canvas setup
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Control frame
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10, fill=tk.X)
        
        # Label buttons
        self.buttons = []
        for idx, (text, label) in enumerate(zip(
            ["Header", "Body", "Footer", "Quote", "Excl."],
            ['Header', 'Body', 'Footer', 'Quote', 'Exclude']
        )):
            btn = tk.Button(self.control_frame, text=text,
                          command=lambda l=label: self.set_current_label(l))
            btn.grid(row=0, column=idx, padx=2)
            self.buttons.append(btn)
            
        # Navigation
        self.next_btn = tk.Button(self.control_frame, text="Next", command=self.next_page,
                                bg="#4CAF50", fg="white")
        self.next_btn.grid(row=0, column=5, padx=10)
        
        # Status
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white")
        self.status_label.pack(pady=5, fill=tk.X)
        
        # Key bindings
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Display settings
        self.zoom = 2
        self.scale = 1.0
        self.geometry_set = False

    def extract_geometric_features(self):
        all_blocks = []
        global_idx = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            raw_blocks = page.get_text("blocks")
            page_blocks = []
            
            for idx, block in enumerate(raw_blocks):
                if len(block) < 6 or not block[4].strip():
                    continue
                
                x0, y0, x1, y1, text = block[:5]
                text = text.strip()
                
                features = {
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                    'width': x1 - x0,
                    'height': y1 - y0,
                    'position': y0 / page.rect.height,
                    'letter_count': calculate_letter_count(text),
                    'font_size': calculate_average_font_size(page, idx),
                    'num_lines': calculate_num_lines(page, idx),
                    'punctuation_proportion': calculate_punctuation_proportion(text),
                    'average_words_per_sentence': calculate_average_words_per_sentence(text),
                    'starts_with_number': calculate_starts_with_number(text),
                    'capitalization_proportion': calculate_capitalization_proportion(text),
                    'average_word_commonality': get_word_commonality(text),
                    'squared_entropy': calculate_entropy(text)**2,
                    'page': page_num,
                    'text': text,
                    'type': '0',
                    'global_idx': global_idx
                }
                page_blocks.append(features)
                global_idx += 1
            
            page_blocks = process_drop_cap(page_blocks)
            all_blocks.extend(page_blocks)
        
        return all_blocks

    def load_current_page(self):
        if self.current_page >= self.total_pages:
            self.finish_classification()
            return
            
        self.current_label = "Body"
        self.update_button_highlight()
        self.current_page_blocks = [b for b in self.all_blocks if b['page'] == self.current_page]
        self.global_indices = [b['global_idx'] for b in self.current_page_blocks]
        
        # Render page image
        page = self.doc.load_page(self.current_page)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Scale to fit window
        max_width = self.root.winfo_screenwidth() * 0.8
        max_height = self.root.winfo_screenheight() * 0.7
        img_w, img_h = img.size
        self.scale = min(max_width/img_w, max_height/img_h, 1)
        new_size = (int(img_w*self.scale), int(img_h*self.scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Initial predictions
        self.update_model_and_predictions()
        self.draw_blocks()
        
        if not self.geometry_set:
            window_height = new_size[1] + 120
            self.root.geometry(f"{new_size[0]}x{window_height}")
            self.geometry_set = True
            
        self.status_var.set(f"Page {self.current_page+1}/{self.total_pages} - Trained on {len(get_training_data()[0])} examples")

    def update_model_and_predictions(self):
        features, labels = get_training_data()
        if features:
            self.mlp_model = train_model(self.mlp_model, features, labels, epochs=15, lr=0.05)
        
        # Predict for current page
        pred_labels = predict_blocks(self.mlp_model, self.current_page_blocks)
        
        # Update only unclassified blocks
        for local_idx, global_idx in enumerate(self.global_indices):
            if self.block_classifications[global_idx] == '0':
                self.block_classifications[global_idx] = pred_labels[local_idx]

    def draw_blocks(self):
        for idx, block in enumerate(self.current_page_blocks):
            global_idx = block['global_idx']
            color = self.get_block_color(global_idx)
            
            x0 = block['x0'] * self.zoom * self.scale
            y0 = block['y0'] * self.zoom * self.scale
            x1 = block['x1'] * self.zoom * self.scale
            y1 = block['y1'] * self.zoom * self.scale
            
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)

    def get_block_color(self, global_idx):
        colors = {
            'Header': '#ff0000',
            'Body': '#00aaff',
            'Footer': '#0000ff',
            'Quote': '#ffff00',
            'Exclude': '#808080',
            '0': 'black'
        }
        return colors.get(self.block_classifications[global_idx], 'black')

    def on_canvas_click(self, event):
        x = event.x / (self.zoom * self.scale)
        y = event.y / (self.zoom * self.scale)
        
        for block in self.current_page_blocks:
            if (block['x0'] <= x <= block['x1'] and 
                block['y0'] <= y <= block['y1']):
                
                global_idx = block['global_idx']
                self.block_classifications[global_idx] = self.current_label
                add_training_example(block, self.current_label)
                
                # Update page buffer
                self.page_buffer.append({
                    'text': block['text'],
                    'label': self.current_label,
                    'y0': block['y0'],
                    'x0': block['x0']
                })
                
                # Immediate update
                self.update_model_and_predictions()
                self.draw_blocks()
                self.status_var.set(f"Page {self.current_page+1}/{self.total_pages} - Trained on {len(get_training_data()[0])} examples")
                break

    def next_page(self):
        # Save current page in reading order
        if self.page_buffer:
            sorted_blocks = sorted(self.page_buffer, key=lambda x: (x['y0'], x['x0']))
            with open("output.txt", "a", encoding="utf-8") as f:
                for block in sorted_blocks:
                    if block['label'] != 'Exclude':
                        f.write(f"[{block['label']}]\n{block['text']}\n\n")
        
        # Reset for next page
        self.page_buffer = []
        self.current_page += 1
        
        if self.current_page >= self.total_pages:
            self.finish_classification()
        else:
            self.load_current_page()

    def set_current_label(self, label):
        self.current_label = label
        self.update_button_highlight()

    def update_button_highlight(self):
        for btn in self.buttons:
            text = btn['text']
            label = 'Exclude' if text == 'Excl.' else text
            btn.config(relief=tk.SUNKEN if label == self.current_label else tk.RAISED)

    def on_key_press(self, event):
        key = event.keysym.lower()
        labels = {'h':'Header', 'b':'Body', 'f':'Footer', 'q':'Quote', 'e':'Exclude'}
        if key in labels:
            self.set_current_label(labels[key])

    def finish_classification(self):
        messagebox.showinfo("Complete", "Classification saved to output.txt!")
        self.doc.close()
        self.root.quit()

    def on_close(self):
        self.doc.close()
        self.root.destroy()

def main():
    file_name = input("Enter PDF filename (without extension): ").strip()
    pdf_path = f"{file_name}.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        return
    
    # Clear output file
    open("output.txt", "w").close()
    
    ManualClassifierGUI(pdf_path)
    print("Classification complete!")

if __name__ == "__main__":
    main()
