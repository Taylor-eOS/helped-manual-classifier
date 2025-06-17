import random
import fitz
import torch
import torch.nn as nn
import torch.optim as optim
from model import BlockClassifier
from utils import extract_page_geometric_features
import numpy as np
import settings

def pre_compute_global_stats(blocks, total_pages):
    fs = [b['font_size'] for b in blocks]
    ws = [b['width'] for b in blocks]
    hs = [b['height'] for b in blocks]
    ls = [b['letter_count'] for b in blocks]
    ps = [b['position'] for b in blocks]
    return {
        'font_size_mean': np.mean(fs), 'font_size_std': np.std(fs),
        'width_mean': np.mean(ws), 'width_std': np.std(ws),
        'height_mean': np.mean(hs), 'height_std': np.std(hs),
        'letter_count_mean': np.mean(ls), 'letter_count_std': np.std(ls),
        'position_mean': np.mean(ps), 'position_std': np.std(ps),
        'total_blocks': len(blocks), 'total_pages': total_pages}

def pre_get_global_features(block, stats, doc_w = 612, doc_h = 792):
    orig = [
        block['odd_even'],
        block['x0'] / doc_w,
        block['y0'] / doc_h,
        block['width'] / doc_w,
        block['height'] / doc_h,
        block['position'],
        block['letter_count'] / 100,
        block['font_size'] / 24,
        block['relative_font_size'],
        block['num_lines'] / 10,
        block['punctuation_proportion'],
        block['average_words_per_sentence'] / 10,
        block['starts_with_number'],
        block['capitalization_proportion'],
        block['average_word_commonality'],
        block['squared_entropy']]
    pct = sum(1 for v in [b['font_size'] for b in stats['blocks']] if v <= block['font_size']) / stats['blocks_count']
    z = (block['font_size'] - stats['font_size_mean']) / (stats['font_size_std'] + 1e-6)
    pg = block['page_num'] / stats['total_pages']
    cc = sum(1 for b in stats['blocks']
             if b['page_num'] != block['page_num'] and
             abs(b['y0'] - block['y0']) < 50 and
             abs(b['font_size'] - block['font_size']) < 2) / (stats['total_pages'] - 1 + 1e-6)
    return orig + [pct, z, pg, cc]

if __name__ == "__main__":
    doc = fitz.open(settings.pretrain_pdf_path)
    all_blocks = []
    for p in range(doc.page_count):
        page_blocks = extract_page_geometric_features(doc, p)
        for b in page_blocks:
            b['page_num'] = p
        all_blocks.extend(page_blocks)
    stats = pre_compute_global_stats(all_blocks, doc.page_count)
    input_dim = len(pre_get_global_features(all_blocks[0], {'blocks': all_blocks, 'blocks_count': len(all_blocks), **stats}))
    model = BlockClassifier(input_features = input_dim).to(settings.device)
    reconstructor = nn.Linear(64, input_dim).to(settings.device)
    opt = optim.AdamW(list(model.parameters()) + list(reconstructor.parameters()), lr = settings.pretrain_learning_rate)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(settings.pretrain_epochs):
        losses = []
        for p in range(doc.page_count):
            page_blocks = extract_page_geometric_features(doc, p)
            for b in page_blocks:
                b['page_num'] = p
            feats = [pre_get_global_features(b, {'blocks': all_blocks, 'blocks_count': len(all_blocks), **stats}) for b in page_blocks]
            if len(feats) < 2:
                continue
            X = torch.tensor(feats, dtype = torch.float32, device = settings.device)
            n = len(feats)
            mcount = max(1, int(n * settings.pretrain_mask_ratio))
            mcount = max(mcount, 1)
            idxs = random.sample(range(n), mcount)
            Xm = X.clone()
            Xm[idxs] = 0
            h = model.relu(model.initial_fc(Xm))
            h = model.resblock1(h)
            h = model.resblock2(h)
            pred = reconstructor(h[idxs])
            tgt = X[idxs]
            l = loss_fn(pred, tgt)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(l.item())
        avg = sum(losses) / len(losses) if losses else 0
        print(f"Epoch {epoch + 1}/{settings.pretrain_epochs} Loss {avg:.4f}")
    torch.save(model.state_dict(), settings.pretrained_file)

