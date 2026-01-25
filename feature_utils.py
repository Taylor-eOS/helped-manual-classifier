import os
import numpy as np
import utils
import settings

class FeatureUtils:
    def compute_global_stats(self):
        if not self.all_blocks:
            self.global_stats = {'features': {}, 'total_blocks': 0, 'total_pages': len(getattr(self, 'doc', []))}
            return
        numeric_values = {}
        for b in self.all_blocks:
            for k, v in b.items():
                if k in ('raw_embedding', 'text'):
                    continue
                if isinstance(v, (list, tuple, np.ndarray)):
                    continue
                try:
                    if isinstance(v, (bool, np.bool_)):
                        numeric_values.setdefault(k, []).append(1.0 if v else 0.0)
                    elif isinstance(v, (int, float, np.number)) and not isinstance(v, (str, bytes)):
                        numeric_values.setdefault(k, []).append(float(v))
                except Exception:
                    continue
        feature_stats = {}
        for k, vals in numeric_values.items():
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                continue
            p_low = float(np.percentile(arr, 0.5))
            p_high = float(np.percentile(arr, 99.5))
            feature_stats[k] = {
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'p005': p_low,
                'p995': p_high
            }
        self.global_stats = {'features': feature_stats, 'total_blocks': len(self.all_blocks), 'total_pages': len(getattr(self, 'doc', []))}

    def dump_block_features(self, feats, names):
        feature_file = settings.feature_data_file
        dirpath = os.path.dirname(feature_file)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        first = not os.path.exists(feature_file) or os.path.getsize(feature_file) == 0
        with open(feature_file, 'a', encoding='utf-8') as f:
            if first:
                f.write(','.join(names) + '\n')
            line = ','.join(f"{x:.3f}" for x in feats) + '\n'
            f.write(line)
        self._dump_counter = self._dump_counter + 1

    def is_consistent_across_pages(self, block):
        if not self.all_blocks:
            return 0.0
        similar = [b for b in self.all_blocks
                   if b['page_num'] != block['page_num']
                   and abs(b['y0'] - block['y0']) < 50
                   and abs(b['font_size'] - block['font_size']) < 2]
        other_pages = {b['page_num'] for b in similar}
        total_other = max(self.global_stats.get('total_pages',1) - 1, 1)
        return len(other_pages) / total_other

    def get_percentile(self, value, all_values):
        if not all_values:
            return 0.5
        return sum(1 for v in all_values if v <= value) / len(all_values)

