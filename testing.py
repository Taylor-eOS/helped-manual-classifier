import os, sys, json
import fitz
import torch
from collections import defaultdict
from main_script import ManualClassifierGUI
from model import BlockClassifier
from utils import extract_page_geometric_features, process_drop_cap
import settings

class PDFEvaluator:
    def __init__(self, pdf_path, ground_truth_path):
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        self.model = BlockClassifier()
        self.gui = ManualClassifierGUI(pdf_path, launch_gui=False)
        self.gui.model = self.model
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
        weights_file = settings.pretrained_file
        if os.path.exists(weights_file):
            self.model.load_state_dict(torch.load(weights_file))
            print(f"Loaded pretrained weights")

    def process_page(self, page_num):
        blocks = extract_page_geometric_features(self.doc, page_num)
        return process_drop_cap(blocks)

    def _predict_blocks(self, blocks, page_num, doc_width=612, doc_height=792):
        if not blocks:
            return []
        for block in blocks:
            block['page_num'] = page_num
        self.model.eval()
        X = torch.tensor([self.gui.get_global_features(b, doc_width, doc_height, False) for b in blocks], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X)
            _, preds = torch.max(logits, 1)
        label_map = ['h1', 'p', 'footer', 'blockquote', 'exclude']
        return [label_map[p] for p in preds.tolist()]

    def evaluate(self):
        total_correct = 0
        total_samples = 0
        testing_label_map = {'h1': 'header', 'p': 'body', 'footer': 'footer', 'blockquote': 'quote', 'exclude': 'exclude'}
        while self.current_page < self.total_pages:
            blocks = self.process_page(self.current_page)
            gt_labels = self.ground_truth.get(self.current_page, [])
            pred_labels = self._predict_blocks(blocks, self.current_page)
            page_true = gt_labels[:len(blocks)]
            page_pred = pred_labels[:len(gt_labels)]
            if page_true:
                correct = sum(t == p for t, p in zip(page_true, page_pred))
                page_accuracy = correct / len(page_true)
                total_correct += correct
                total_samples += len(page_true)
                cumulative_accuracy = total_correct / total_samples
                print(f"Page {self.current_page + 1}: Page accuracy {page_accuracy:.2%}, cumulative accuracy {cumulative_accuracy:.2%}")
                if settings.print_mistaken_predictions:
                    for i, (true, pred) in enumerate(zip(page_true, page_pred)):
                        if pred != true:
                            if settings.print_predictions:
                                print(f"Block {i}: {pred} - {true}")
                            with open('mistaken_predictions.txt', 'a') as f:
                                f.write(f"Page {self.current_page + 1}, block {i}: {pred} - {true}\n")
            for block, true_label in zip(blocks, gt_labels):
                self.gui.add_training_example(block, testing_label_map[true_label])
            if self.gui.training_data:
                features, labels = zip(*self.gui.training_data)
                self.gui.training_data.clear()
                self.model = self.gui.train_model(features, labels, False)
            self.current_page += 1
        self.doc.close()
        final_accuracy = total_correct / total_samples if total_samples else 0
        print(f"\nFinal Accuracy: {final_accuracy:.2%}")
        return final_accuracy

def main():
    if len(sys.argv) != 2:
        print("Usage: add input pdf file basename as only argument")
        sys.exit(1)
    name = sys.argv[1]
    evaluator = PDFEvaluator(f"{name}.pdf", "ground_truth.json")
    final_acc = evaluator.evaluate()
    if settings.save_testing_weights: torch.save(evaluator.model.state_dict(), 'testing_weights.pth')

if __name__ == "__main__":
    if settings.print_mistaken_predictions:
        with open('mistaken_predictions.txt', 'w') as f:
            f.write(f"Page, block: predicted - true\n")
    main()

