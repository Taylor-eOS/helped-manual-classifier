import random
import fitz
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import BlockClassifier
from utils import extract_page_geometric_features
import settings

def pre_compute_global_stats(blocks, total_pages):
    fs = [b['font_size'] for b in blocks]
    stats = {
        'blocks': blocks,
        'blocks_count': len(blocks),
        'total_pages': total_pages,
        'font_size_mean': np.mean(fs),
        'font_size_std': np.std(fs)}
    return stats

def pre_get_global_features(block, stats, doc_w=612, doc_h=792):
    orig = []
    for name in utils.get_base_features():
        value = block[name]
        scale = settings.SCALES.get(name)
        if scale == 'doc_width':
            value /= doc_w
        elif scale == 'doc_height':
            value /= doc_h
        elif isinstance(scale, (int, float)):
            value /= scale
        orig.append(value)
    pct = sum(1 for b in stats['blocks']
              if b['font_size'] <= block['font_size']
             ) / stats['blocks_count']
    z = (block['font_size'] - stats['font_size_mean']) / (stats['font_size_std'] + 1e-6)
    pg = block['page_num'] / stats['total_pages']
    cc = sum(
        1 for b in stats['blocks']
        if b['page_num'] != block['page_num']
        and abs(b['y0'] - block['y0']) < 50
        and abs(b['font_size'] - block['font_size']) < 2
    ) / (stats['total_pages'] - 1 + 1e-6)
    return orig + [pct, z, pg, cc]

