import os
import fitz
import string
import numpy as np
from math import log2
from collections import Counter
from wordfreq import word_frequency

def extract_blocks(pdf_path):
    blocks = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        for block in page.get_text("blocks"):
            blocks.append({
                'page': page_num,
                'x0': block[0],
                'y0': block[1],
                'x1': block[2],
                'y1': block[3],
                'raw_block': block
            })
    doc.close()
    return blocks

def drop_to_file(block_text, block_type, block_page_number):
    with open("output.txt", "a", encoding='utf-8') as file:
        if block_type == 'Header':
            file.write(f"<h1>{block_text}</h1>\n<{block_page_number + 1}>\n\n")
        elif block_type == 'Body':
            file.write(f"<body>{block_text}</body>\n<{block_page_number + 1}>\n\n")
        elif block_type == 'Footer':
            file.write(f"<footer>{block_text}</footer>\n<{block_page_number + 1}>\n\n")
        elif block_type == 'Quote':
            file.write(f"<blockquote>{block_text}</blockquote>\n<{block_page_number + 1}>\n\n")
        else:
            file.write(f"{block_text} ERROR\n\n")

def delete_if_exists(del_file):
    if os.path.exists(del_file):
        os.remove(del_file)

# Feature calculation functions
def calculate_height(y0, y1):
    return y1 - y0

def calculate_width(x0, x1):
    return x1 - x0

def calculate_position(y0, page_height):
    return y0 / page_height

def calculate_letter_count(text):
    #return sum(c.isalpha() for c in text)
    return sum(c.isalpha() or c.isnumeric() for c in text)

def calculate_punctuation_proportion(text):
    total = len(text)
    return sum(1 for c in text if c in string.punctuation) / total if total > 0 else 0

def calculate_average_font_size(page, block_index):
    try:
        block = page.get_text("dict")["blocks"][block_index]
        spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
        return sum(span["size"] for span in spans) / len(spans) if spans else 12
    except:
        return 12

def calculate_num_lines(page, block_index):
    try:
        return len(page.get_text("dict")["blocks"][block_index].get("lines", []))
    except:
        return 1

def calculate_average_words_per_sentence(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

def calculate_starts_with_number(text): 
    return int(text.strip()[0].isdigit()) if text.strip() else 0

def calculate_capitalization_proportion(text):
    alpha = sum(c.isalpha() for c in text)
    return sum(c.isupper() for c in text) / alpha if alpha > 0 else 0

def get_word_commonality(text):
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.isalpha()]
    if not words: return 0.01
    freqs = [word_frequency(w, 'en') for w in words]
    return np.mean([f for f in freqs if f > 0]) * 100 if any(f > 0 for f in freqs) else 0.01

def calculate_entropy(text):
    if not text: return 0
    text = text.lower()
    counts = np.array([text.count(c) for c in set(text)])
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def process_drop_cap(page_data):
    font_sizes = [block['font_size'] for block in page_data]
    if not font_sizes:
        return page_data
    avg_size = np.mean(font_sizes)
    std_dev = np.std(font_sizes)
    threshold = avg_size + 2 * std_dev
    for i, block in enumerate(page_data):
        if block['font_size'] > threshold and block['letter_count'] < 10:
            if i + 1 < len(page_data):
                page_data[i]['font_size'] = page_data[i+1]['font_size']
    max_size = max(font_sizes)
    for block in page_data:
        block['relative_font_size'] = block['font_size'] / max_size
    return page_data

