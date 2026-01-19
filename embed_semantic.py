import os, threading
import torch
import numpy as np
from embed import _load_components
import settings

def get_raw_embedding(texts, batch_size=64):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, 384))
    tokenizer, model = _load_components("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    device = settings.device
    model.to(device)
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t[:settings.truncate_embedding_input] for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hs = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        summed = (hs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_vecs.append(normed.cpu().numpy())
    return np.vstack(all_vecs)
