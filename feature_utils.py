import numpy as np
import settings

class FeatureUtils:
    def compute_global_stats(self): #features_here
        if not self.all_blocks:
            return
        fs = [b['font_size'] for b in self.all_blocks]
        ws = [b['width']     for b in self.all_blocks]
        hs = [b['height']    for b in self.all_blocks]
        ls = [b['letter_count'] for b in self.all_blocks]
        ps = [b['position']  for b in self.all_blocks]
        tp = len(self.all_blocks)
        pg = len(self.doc)
        m, s = np.mean, np.std
        self.global_stats = {
            'font_size_mean': m(fs), 'font_size_std': s(fs),
            'width_mean': m(ws),     'width_std': s(ws),
            'height_mean': m(hs),    'height_std': s(hs),
            'letter_count_mean': m(ls), 'letter_count_std': s(ls),
            'position_mean': m(ps),  'position_std': s(ps),
            'total_blocks': tp,      'total_pages': pg}

    def get_percentile(self, value, all_values):
        if not all_values:
            return 0.5
        return sum(1 for v in all_values if v <= value) / len(all_values)

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

    def get_global_features(self, block, doc_width = 612, doc_height = 792, dump = False):
        orig = []
        for name in settings.BASE_FEATURES:
            v = block[name]
            scale = settings.SCALES.get(name)
            if isinstance(scale, str):
                v /= locals()[scale]
            elif scale:
                v /= scale
            orig.append(v)
        if self.global_stats:
            p = self.get_percentile(block['font_size'], [b['font_size'] for b in self.all_blocks])
            z = (block['font_size'] - self.global_stats['font_size_mean']) / (self.global_stats['font_size_std'] + 1e-6)
            pg = block['page_num'] / self.global_stats['total_pages']
            c = self.is_consistent_across_pages(block)
            glob = [p, z, pg, c]
        else:
            glob = [0.5, 0.0, 0.5, 0.0]
        feats = orig + glob
        if dump:
            self.dump_block_features(orig, glob)
        return feats

    def dump_block_features(self, orig, glob): #features_here
        if not hasattr(self, '_dumped_signatures'):
            self._dumped_signatures = set()
        sig = tuple(orig)
        if sig in self._dumped_signatures:
            return
        feats = orig + glob
        names = settings.BASE_FEATURES + ['font_size_percentile', 'font_size_zscore', 'page_fraction', 'cross_page_consistency']
        try:
            with open(settings.feature_data_file, "a") as f:
                if f.tell() == 0:
                    f.write("Block," + ",".join(names) + "\n")
                idx = len(self._dumped_signatures)
                f.write(f"{idx}," + ",".join(f"{x:.5f}" for x in feats) + "\n")
        except Exception as e:
            print(f"Dump failed: {e}")
        self._dumped_signatures.add(sig)

