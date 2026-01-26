import numpy as np

def build_orig_features(block, doc_width, doc_height, base_features, scales, per_page_stats):
    orig = []
    orig_names = []
    pn = block.get('page_num', -1)
    pstats = per_page_stats.get(pn, {}) if per_page_stats else {}
    for name in base_features:
        v = block.get(name, 0.0)
        if name in pstats:
            min_v, rng = pstats[name]
            v = (v - min_v) / rng
        else:
            scale = scales.get(name)
            if isinstance(scale, str):
                denom = {'doc_width': doc_width, 'doc_height': doc_height}.get(scale, 1.0)
                if denom != 1.0:
                    v = v / denom
            elif isinstance(scale, (int, float)) and scale != 0.0:
                v = v / scale
        orig.append(float(v))
        orig_names.append(name)
    return orig, orig_names

def build_global_stat_features(block, all_blocks, global_stats, get_percentile_fn, is_consistent_fn):
    if global_stats and 'font_size' in block:
        fs = block.get('font_size', 0.0)
        all_fs = [b.get('font_size', 0.0) for b in all_blocks]
        p = get_percentile_fn(fs, all_fs)
        mean = global_stats['features'].get('font_size', {}).get('mean', 0.0)
        std = global_stats['features'].get('font_size', {}).get('std', 1.0)
        z = (fs - mean) / (std + 1e-6)
        z = max(min(z, 5.0), -5.0)
        z = (z + 5.0) / 10.0
        pg = block.get('page_num', 0) / max(1, global_stats.get('total_pages', 1))
        c = float(is_consistent_fn(block))
        glob = [p, z, pg, c]
    else:
        glob = [0.0, 0.0, 0.0, 0.0]
    glob_names = ['font_size_percentile', 'font_size_zscore', 'page_frac', 'consistency']
    return glob, glob_names

def build_semantic_features(block, semantic_override, get_semantic_logits_fn):
    if semantic_override is not None:
        semantic_conf = list(semantic_override)
    else:
        emb = block.get('raw_embedding', [0.0] * 384)
        _, probs = get_semantic_logits_fn(emb)
        semantic_conf = probs.tolist()
    semantic_conf = semantic_conf[:5]
    if len(semantic_conf) != 5:
        raise ValueError(f"Semantic head returned {len(semantic_conf)} classes, expected 5")
    semantic_names = [f'semantic_{i}' for i in range(len(semantic_conf))]
    return semantic_conf, semantic_names

