import os
import fitz
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from utils import extract_blocks, drop_to_file
from mlp_model import BlockClassifier, train_model, predict_blocks, get_training_data

class ManualClassifierGUI:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.all_blocks = extract_blocks(pdf_path)
        self.current_page = 0
        self.block_classifications = ['0'] * len(self.all_blocks)
        self.current_label = 'Body'
        self.pending_classification = None
        self.label_colors = {
            'Header': '#ff0000',
            'Body': '#00aaff',
            'Footer': '#0000ff',
            'Quote': '#ffff00',
            'Exclude': '#808080',
            '0': 'black'
        }
        self.key_to_label = {
            'h': 'Header',
            'b': 'Body',
            'f': 'Footer',
            'q': 'Quote',
            'e': 'Exclude'
        }
        self.mlp_model = BlockClassifier()
        self.root = tk.Tk()
        self.root.title("PDF Block Classifier with MLP Preselection")
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10, fill=tk.X)
        self.button_texts = ["Header", "Body", "Footer", "Quote", "Excl."]
        self.buttons = []
        for idx, text in enumerate(self.button_texts):
            btn = tk.Button(self.control_frame, text=text,
                            command=lambda t=text: self.set_current_label(t if t != "Excl." else "Exclude"))
            btn.grid(row=0, column=idx, padx=1)
            self.buttons.append(btn)
        self.next_button = tk.Button(self.control_frame, text="Next", command=self.next_page,
                                     bg="#4CAF50", fg="white")
        self.next_button.grid(row=0, column=len(self.button_texts), padx=1)
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white")
        self.status_label.pack(pady=5, fill=tk.X)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.zoom = 2
        self.scale = 1.0
        self.geometry_set = False
        self.load_current_page()
        self.root.mainloop()

    def set_current_label(self, label):
        self.current_label = label
        self.update_button_highlight()

    def update_button_highlight(self):
        for btn in self.buttons:
            btn.config(relief=tk.SUNKEN if btn['text'] ==
                       (self.current_label if self.current_label != "Exclude" else "Excl.") else tk.RAISED)

    def load_current_page(self):
        if self.current_page >= self.total_pages:
            self.finish_classification()
            return
        self.current_label = "Body"
        self.update_button_highlight()
        page = self.doc.load_page(self.current_page)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        max_width = self.root.winfo_screenwidth() * 0.8
        max_height = self.root.winfo_screenheight() * 0.8
        img_width, img_height = img.size
        self.scale = min(max_width / img_width, max_height / img_height, 1)
        new_size = (int(img_width * self.scale), int(img_height * self.scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.apply_model_predictions()
        for idx, block in enumerate(self.all_blocks):
            if block['page'] != self.current_page:
                continue
            zx0 = block['x0'] * self.zoom * self.scale
            zy0 = block['y0'] * self.zoom * self.scale
            zx1 = block['x1'] * self.zoom * self.scale
            zy1 = block['y1'] * self.zoom * self.scale
            color = self.label_colors.get(self.block_classifications[idx], 'black')
            self.canvas.create_rectangle(zx0, zy0, zx1, zy1, outline=color, fill="", width=2)
        if not self.geometry_set:
            window_height = new_size[1] + 120
            self.root.geometry(f"{new_size[0]}x{window_height}")
            self.geometry_set = True
        self.status_var.set(f"Page {self.current_page + 1} of {self.total_pages}")
        self.update_button_highlight()

    def apply_model_predictions(self):
        indices = [i for i, block in enumerate(self.all_blocks) if block['page'] == self.current_page]
        page_blocks = [self.all_blocks[i] for i in indices]
        train_features, train_labels = get_training_data(page_blocks)
        if len(train_features) > 0:
            self.mlp_model = train_model(self.mlp_model, train_features, train_labels, epochs=100, lr=0.01)
        pred_labels = predict_blocks(self.mlp_model, page_blocks)
        for j, i in enumerate(indices):
            self.block_classifications[i] = pred_labels[j]

    def get_block_text(self, block):
        text = block.get("text", "").strip()
        if text:
            return text
        if "raw_block" in block and len(block["raw_block"]) >= 5:
            return block["raw_block"][4].strip()
        return ""

    def on_canvas_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        pdf_x = x / (self.zoom * self.scale)
        pdf_y = y / (self.zoom * self.scale)
        for idx, block in enumerate(self.all_blocks):
            if block['page'] != self.current_page:
                continue
            if block['x0'] <= pdf_x <= block['x1'] and block['y0'] <= pdf_y <= block['y1']:
                if self.pending_classification is not None:
                    drop_to_file(self.pending_classification[0],
                                 self.pending_classification[1],
                                 self.pending_classification[2])
                    self.pending_classification = None
                text = self.get_block_text(block)
                self.pending_classification = (text, self.current_label, block.get("page", self.current_page))
                self.block_classifications[idx] = self.current_label
                self.all_blocks[idx]["type"] = self.current_label
                self.redraw_page()
                break

    def redraw_page(self):
        page = self.doc.load_page(self.current_page)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        max_width = self.root.winfo_screenwidth() * 0.8
        max_height = self.root.winfo_screenheight() * 0.8
        img_width, img_height = img.size
        self.scale = min(max_width / img_width, max_height / img_height, 1)
        new_size = (int(img_width * self.scale), int(img_height * self.scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        for idx, block in enumerate(self.all_blocks):
            if block['page'] != self.current_page:
                continue
            zx0 = block['x0'] * self.zoom * self.scale
            zy0 = block['y0'] * self.zoom * self.scale
            zx1 = block['x1'] * self.zoom * self.scale
            zy1 = block['y1'] * self.zoom * self.scale
            color = self.label_colors.get(self.block_classifications[idx], 'black')
            self.canvas.create_rectangle(zx0, zy0, zx1, zy1, outline=color, fill="", width=2)
        self.update_button_highlight()

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in self.key_to_label:
            self.set_current_label(self.key_to_label[key])

    def process_current_page(self):
        if self.pending_classification is not None:
            drop_to_file(self.pending_classification[0],
                         self.pending_classification[1],
                         self.pending_classification[2])
            self.pending_classification = None
        indices = [i for i, block in enumerate(self.all_blocks) if block['page'] == self.current_page]
        # Sort by y0 then x0 (natural reading order)
        sorted_blocks = sorted([(i, self.all_blocks[i]) for i in indices],
                               key=lambda tup: (tup[1].get('y0', 0), tup[1].get('x0', 0)))
        for i, block in sorted_blocks:
            classification = self.block_classifications[i] or 'Exclude'
            if classification != 'Exclude':
                text = self.get_block_text(block)
                drop_to_file(text, classification, block.get("page", 0))

    def next_page(self):
        self.process_current_page()
        self.current_page += 1
        if self.current_page >= self.total_pages:
            self.finish_classification()
        else:
            self.load_current_page()

    def finish_classification(self):
        if self.current_page < self.total_pages:
            self.process_current_page()
        messagebox.showinfo("Complete", "Classification saved successfully!")
        self.doc.close()
        self.root.quit()

    def on_close(self):
        self.root.destroy()

def main():
    file_name = input("Enter PDF file name (without extension): ").strip()
    pdf_path = f"{file_name}.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found!")
        return
    open("output.txt", "w", encoding='utf-8').close()
    ManualClassifierGUI(pdf_path)
    print("Classification complete!")

if __name__ == "__main__":
    main()

