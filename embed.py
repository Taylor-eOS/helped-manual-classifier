import threading
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import settings

_model_lock = threading.Lock()
_model = None
_tokenizer = None

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def _load_components(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print("Loading embedding model")
    global _model, _tokenizer
    with _model_lock:
        if _model is None or _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModel.from_pretrained(model_name)
    return _tokenizer, _model

def get_embedding(texts, batch_size=32):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    tokenizer, model = _load_components()
    device = settings.device
    model.to(device)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t[:settings.truncate_embedding_input] for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        pooled = _mean_pooling(out, enc["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
    raw = np.vstack(all_embeddings)
    reduced = apply_document_pca(raw, settings.embedding_components)
    return raw

def apply_document_pca(raw_embeddings, n_components_desired):
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
        "I would be happy to do this",
        "Certainly",
        "He rejoiced with anticipation",
        "Refusing is the best option",
        "This is not ok",
        "All bets are off",]
    vectors = get_embedding(test_sentences)
    for i, vec in enumerate(vectors):
        print(f"{i}: {', '.join(f'{v:.4f}' for v in vec)}")

