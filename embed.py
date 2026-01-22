import os, threading
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import settings

_model_lock = threading.Lock()
_model = None
_tokenizer = None

def _load_components(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print("Loading embedding model")
    global _model, _tokenizer
    with _model_lock:
        if _model is None or _tokenizer is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            _model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    return _tokenizer, _model

def get_embedding(texts, batch_size):
    if settings.embedding_components == 0:
        return np.zeros((len(texts), 0))
    from sklearn.decomposition import TruncatedSVD
    if settings.use_jina:
        return get_embedding_jina(texts, batch_size)
    else:
        return get_embedding_st(texts, batch_size)

def get_embedding_jina(texts, batch_size=64):
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    device = settings.device; model.to(device)
    if isinstance(texts, str): texts = [texts]
    if not texts: return np.zeros((0, settings.embedding_components))
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t[: settings.truncate_embedding_input] for t in batch]
        emb = model.encode(batch, task="classification", truncate_dim=32)
        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        all_vecs.append(normed)
    return np.vstack(all_vecs)

def get_embedding_st(texts, batch_size=64):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, settings.embedding_components))
    tokenizer, model = _load_components()
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
        cls = hs[:, 0]
        scores = torch.einsum('bth,bh->bt', hs, cls)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pooled = (hs * weights).sum(dim=1)
        normed = torch.nn.functional.normalize(attn_pooled, p=2, dim=1)
        all_vecs.append(normed.cpu().numpy())
    raw = np.vstack(all_vecs)
    svd = TruncatedSVD(n_components=settings.embedding_components)
    return svd.fit_transform(raw)
    #return apply_document_pca(raw)

def apply_document_pca(raw_embeddings, n_components_desired=settings.embedding_components):
    from sklearn.decomposition import PCA
    if len(raw_embeddings) == 0:
        return np.zeros((0, n_components_desired))
    n_components = min(n_components_desired, raw_embeddings.shape[0])
    pca = PCA(n_components=n_components, whiten=True)
    transformed = pca.fit_transform(raw_embeddings)
    if transformed.shape[1] < n_components_desired:
        padding = np.zeros((len(raw_embeddings), n_components_desired - n_components))
        return np.hstack([transformed, padding])
    return transformed

if __name__ == "__main__":
    test_sentences = [
        "I would be happy to comply.",
        "Certainly.",
        "Refusing is the safer option.",
        "No.",]
    vectors = get_embedding(test_sentences)
    vectors = apply_document_pca(vectors)
    for i, vec in enumerate(vectors):
        print(f"{i}: {', '.join(f'{v:.4f}' for v in vec)}")

