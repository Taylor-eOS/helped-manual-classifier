import os, json, string
import fitz
import numpy as np
from math import log2
from collections import Counter
from wordfreq import word_frequency
import settings

def drop_to_file(block_text, block_type, block_page_number):
    if settings.debug: print(type(block_text), type(block_type), type(block_page_number), sep='\n', end='\n')
    label_mapping = {"header": "h1", "body": "p", "footer": "footer", "quote": "blockquote", "exclude": "exclude"}
    entry = {
        "label": label_mapping.get(block_type, "unknown"),
        "page": block_page_number + 1,
        "text": block_text}
    with open("output.json", "a", encoding='utf-8') as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if settings.ground_truth_logging:
            entry['label'] = block_type
            if block_type == "exclude":
                entry['text'] = ""
            with open("ground_truth.json", "a", encoding='utf-8') as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if settings.debug: print(entry)

def extract_page_geometric_features(doc, page_num):
    page = doc.load_page(page_num)
    page_height = page.rect.height
    raw_blocks = page.get_text("blocks")
    page_blocks = []
    for idx, block in enumerate(raw_blocks):
        if len(block) < 6 or not block[4].strip():
            continue
        x0, y0, x1, y1, text = block[:5]
        text = text.strip()
        page_blocks.append({
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'width': x1 - x0, 'height': y1 - y0,
            'position': y0 / page_height,
            'letter_count': calculate_letter_count(text),
            'font_size': calculate_average_font_size(page, idx),
            'relative_font_size': None,
            'num_lines': calculate_num_lines(page, idx),
            'punctuation_proportion': calculate_punctuation_proportion(text),
            'average_words_per_sentence': calculate_average_words_per_sentence(text),
            'starts_with_number': calculate_starts_with_number(text),
            'capitalization_proportion': calculate_capitalization_proportion(text),
            'average_word_commonality': get_word_commonality(text),
            'squared_entropy': calculate_entropy(text)**2,
            'page_num': page_num,
            'odd_even': 1 if page_num % 2 == 0 else 0,
            'text': text, 'type': '0'
        })
    page_blocks = sorted(page_blocks, key=lambda b: b['y0'])
    page_blocks = add_vertical_neighbour_distances(page_blocks, page_height)
    page_blocks = process_drop_cap(page_blocks)
    return page_blocks

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

def get_base_features():
    base = settings.BASE_FEATURES
    base += [f'embed_{i}' for i in range(settings.embedding_components)]
    return base

###Feature calculation functions
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

def add_vertical_neighbour_distances(blocks, page_height):
    if not blocks:
        return blocks
    sentinel = page_height * 0.2
    for i, block in enumerate(blocks):
        if i == 0:
            dist_prev = sentinel
        else:
            prev = blocks[i - 1]
            dist_prev = block['y0'] - prev['y1']
        if i == len(blocks) - 1:
            dist_next = sentinel
        else:
            next_block = blocks[i + 1]
            dist_next = next_block['y0'] - block['y1']
        block['dist_prev_norm'] = dist_prev / page_height
        block['dist_next_norm'] = dist_next / page_height
    return blocks

