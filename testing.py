import os, sys, json
import fitz
import torch
import numpy as np
from collections import defaultdict
from main_script import ManualClassifierGUI
from model import BlockClassifier
from utils import extract_page_geometric_features, process_drop_cap
from embed_semantic import get_raw_embedding
import settings

class PDFEvaluator:
    def __init__(self, pdf_path, ground_truth_path):
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        self.gui = ManualClassifierGUI(pdf_path, launch_gui=False)
        self.gui.build_models()
        if settings.load_pretraining_weights:
            self.load_model_weights()
        if settings.load_pretraining_weights:
            self.load_model_weights()
        self.current_page = 0

    def load_ground_truth(self, gt_path):
        truth = defaultdict(list)
        with open(gt_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                pnum = entry['page'] - 1
                truth[pnum].append(entry['label'])
        return truth

    def load_model_weights(self):
        if os.path.exists(settings.pretrained_layout_file):
            self.gui.layout_model.load_state_dict(torch.load(settings.pretrained_layout_file))
            print("Loaded pretrained layout weights")
        if os.path.exists(settings.pretrained_semantic_file):
            self.gui.semantic_head.load_state_dict(torch.load(settings.pretrained_semantic_file))
            print("Loaded pretrained semantic weights")

    def process_page(self, page_num):
        blocks = extract_page_geometric_features(self.doc, page_num)
        return process_drop_cap(blocks)

    def _predict_blocks(self, blocks, page_num, doc_width=612, doc_height=792):
        if not blocks:
            return []
        for block in blocks:
            block['page_num'] = page_num
        self.gui.layout_model.eval()
        self.gui.semantic_head.eval()
        feats = []
        for b in blocks:
            emb = b.get('raw_embedding', [0.0] * 384)
            _, probs = self.gui.get_semantic_logits(emb)
            f = self.gui.get_global_features(b, doc_width, doc_height, False, semantic_override=probs)
            feats.append(f)
        X = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            logits = self.gui.layout_model(X)
            _, preds = torch.max(logits, 1)
        label_map = ['h1', 'p', 'footer', 'blockquote', 'exclude']
        return [label_map[p] for p in preds.tolist()]

    def _compile_all_blocks(self):
        all_blocks = []
        all_texts = []
        for p in range(self.total_pages):
            page_blks = self.process_page(p)
            for b in page_blks:
                b['page_num'] = p
            all_blocks.extend(page_blks)
            all_texts.extend([b['text'] for b in page_blks])
        return all_blocks, all_texts

    def _attach_embeddings(self, blocks, texts):
        if texts:
            texts_length = len(texts)
            print(f"Creating {texts_length} embeddings")
            emb = get_raw_embedding(texts, texts_length)
        else:
            emb = np.zeros((0, 384))
        for i, b in enumerate(blocks):
            raw = emb[i] if i < emb.shape[0] else np.zeros(384)
            b['raw_embedding'] = raw.tolist()

    def _page_slices(self, blocks):
        idx = 0
        page_index = {}
        for p in range(self.total_pages):
            cnt = len(self.process_page(p))
            page_index[p] = blocks[idx : idx + cnt]
            idx += cnt
        return page_index

    def evaluate(self):
        all_blocks, all_texts = self._compile_all_blocks()
        self._attach_embeddings(all_blocks, all_texts)
        page_index = self._page_slices(all_blocks)
        total_correct = 0
        total_samples = 0
        error_counts = defaultdict(int)
        true_counts = defaultdict(int)
        testing_label_map = {'h1': 'header', 'p': 'body', 'footer':'footer', 'blockquote':'quote', 'exclude':'exclude'}
        while self.current_page < self.total_pages:
            blocks = page_index[self.current_page]
            gt = self.ground_truth.get(self.current_page, [])
            preds = self._predict_blocks(blocks, self.current_page)
            page_true = gt[:len(blocks)]
            page_pred = preds[:len(gt)]
            limit = min(len(page_true), len(page_pred), len(blocks))
            if limit > 0:
                correct = sum(page_true[i] == page_pred[i] for i in range(limit))
                total_correct += correct
                total_samples += limit
                for i in range(limit):
                    true_counts[page_true[i]] += 1
                page_accuracy = correct / limit
                cumulative_accuracy = total_correct / total_samples
                print(f"Page {self.current_page+1}: Page accuracy {page_accuracy:.2%}, cumulative accuracy {cumulative_accuracy:.2%}")
                if settings.print_mistaken_predictions:
                    self.write_mistaken_predictions(self.current_page+1, page_true, page_pred, blocks, limit, error_counts)
            for block, true_label in zip(blocks, gt):
                self.gui.add_training_example(block, testing_label_map[true_label])
            self.gui.schedule_retrainer()
            for _ in range(self.gui.page_retrain_limit + self.gui.replay_retrain_limit):
                self.gui.retrain_tick()
            self.current_page += 1
        self.doc.close()
        final_acc = total_correct / total_samples if total_samples else 0
        print(f"\nFinal Accuracy: {final_acc:.2%}")
        if settings.print_mistaken_predictions:
            with open('mistaken_predictions.txt', 'a') as f:
                f.write("\nMistake counts:\n")
                for (pred, true), cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
                    total = true_counts[true]
                    prop = cnt / total
                    f.write(f"{true}-{pred}: {cnt}/{total} ({cnt/total*100:.0f}%)\n")
        torch.save(self.gui.layout_model.state_dict(), 'weights_layout.pt')
        torch.save(self.gui.semantic_head.state_dict(), 'weights_semantic.pt')
        return final_acc

    def write_mistaken_predictions(self, page_number, page_true, page_pred, blocks, limit, error_counts):
        for i in range(limit):
            true = page_true[i]
            pred = page_pred[i]
            if pred != true:
                snippet = blocks[i].get('text', '')[:30].replace('\n', '\\n')
                msg = f"Page {page_number}, block {i}: {pred}-{true}   ({snippet})"
                if settings.print_predictions:
                    print(msg)
                with open('mistaken_predictions.txt', 'a') as f:
                    f.write(msg + '\n')
                error_counts[(pred, true)] += 1

def main():
    open(settings.feature_data_file, "w").close()
    #if len(sys.argv) != 2:
    #    print("Usage: add input pdf file basename as only argument")
    #    sys.exit(1)
    #name = sys.argv[1]
    evaluator = PDFEvaluator(f"{settings.test_file_basename}.pdf", "ground_truth.json")
    final_acc = evaluator.evaluate()

if __name__ == "__main__":
    if settings.print_mistaken_predictions:
        open('mistaken_predictions.txt', "w").close()
    main()

