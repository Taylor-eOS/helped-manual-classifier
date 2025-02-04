import fitz
from PIL import Image, ImageTk

def load_page_image(doc, page_num, zoom, root):
    """
    Loads and resizes the page image.
    Returns (photo, new_size, scale) where photo is a Tkinter PhotoImage.
    """
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

def draw_blocks(canvas, blocks, block_classifications, page_num, zoom, scale, label_colors):
    """
    Draw rectangles on the canvas for all blocks on page_num, using the provided colors.
    """
    for idx, block in enumerate(blocks):
        if block.get('page') != page_num:
            continue
        zoomed_x0 = block.get('x0', 0) * zoom * scale
        zoomed_y0 = block.get('y0', 0) * zoom * scale
        zoomed_x1 = block.get('x1', 0) * zoom * scale
        zoomed_y1 = block.get('y1', 0) * zoom * scale
        color = label_colors.get(block_classifications[idx], 'black')
        canvas.create_rectangle(zoomed_x0, zoomed_y0, zoomed_x1, zoomed_y1,
                                outline=color, fill="", width=2)

