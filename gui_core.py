import fitz
import tkinter as tk
from PIL import Image, ImageTk

def load_page_image(doc, page_num, zoom, root):
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    max_width = root.winfo_screenwidth() * 0.8
    max_height = root.winfo_screenheight() * 0.8
    img_width, img_height = img.size
    scale = min(max_width / img_width, max_height / img_height, 1)
    new_size = (int(img_width * scale), int(img_height * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    return photo, new_size, scale

def draw_blocks(self):
    for block in self.current_page_blocks:
        global_idx = block['global_idx']
        color = get_block_color(self.block_classifications[global_idx])
        x0 = block['x0'] * self.zoom * self.scale
        y0 = block['y0'] * self.zoom * self.scale
        x1 = block['x1'] * self.zoom * self.scale
        y1 = block['y1'] * self.zoom * self.scale
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=4)

def get_block_color(global_idx):
    colors = {'header': '#ff0000','body': '#00aaff','footer': 'DarkSeaGreen3','quote': 'gold','exclude': 'gray75','0': 'white'}
    return colors.get(global_idx, 'white')

def update_button_highlight(buttons, current_label):
    for btn in buttons:
        text = btn['text']
        label = 'exclude' if text == 'Excl.' else text.lower()
        btn.config(relief=tk.SUNKEN if label == current_label else tk.RAISED)

def load_current_page(self):
    if self.current_page >= self.total_pages:
        self.finish_classification()
        return
    with self.processing_lock:
        if self.all_blocks[self.current_page] is None:
            self.process_page(self.current_page)
        self.current_page_blocks = self.all_blocks[self.current_page]
        self.global_indices = [b['global_idx'] for b in self.current_page_blocks]
        self._sorted_blocks = sorted(self.current_page_blocks, key=lambda b: (b['y0'], b['x0']))
        self._key_block_idx = 0
        page = self.doc.load_page(self.current_page)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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
        self.update_model_and_predictions()
        self.draw_blocks()

def classify_next_block(self, label_idx):
    if not hasattr(self, "_sorted_blocks"):
        return None
    if self._key_block_idx >= len(self._sorted_blocks):
        return "PAGE_DONE"
    block = self._sorted_blocks[self._key_block_idx]
    label = ['header','body','footer','quote','exclude'][label_idx]
    self.block_classifications[block['global_idx']] = label
    self.page_buffer.append({
        'text': block['text'],
        'label': label,
        'y0': block['y0'],
        'x0': block['x0'],
        'global_idx': block['global_idx']})
    self.draw_blocks()
    self._key_block_idx += 1
    if self._key_block_idx >= len(self._sorted_blocks):
        return "PAGE_DONE"
    return block, label

