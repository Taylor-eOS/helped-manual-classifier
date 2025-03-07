import os
import json
import fitz
import torch
import sys
from collections import defaultdict
from model_util import BlockClassifier, train_model, predict_blocks, add_training_example, get_training_data, training_data, normalization_buffer, EPOCHS, LEARNING_RATE
from utils import extract_page_geometric_features, process_drop_cap

print_predictions = True

class PDFEvaluator:
    def __init__(self, pdf_path, ground_truth_path):
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        self.model = BlockClassifier()
        self.load_model_weights()
        self.current_page = 0  #0-based index

    def load_ground_truth(self, gt_path):
        truth = defaultdict(list)
        with open(gt_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                pnum = entry['page'] - 1  #Convert to 0-based
                truth[pnum].append(entry['label'])
        return truth

    def process_page(self, page_num):
        blocks = extract_page_geometric_features(self.doc, page_num)
        return process_drop_cap(blocks)

    def evaluate(self):
        total_correct = 0
        total_samples = 0
        retrogressive_label_map = {'h1': 'header', 'p': 'body', 'footer': 'footer', 'blockquote': 'quote', 'exclude': 'exclude'}
        while self.current_page < self.total_pages:
            #Process current page
            blocks = self.process_page(self.current_page)
            gt_labels = self.ground_truth.get(self.current_page, [])
            #Predict with current model state
            pred_labels = predict_blocks(self.model, blocks)
            #Calculate accuracy for current page
            page_true = gt_labels[:len(blocks)]
            page_pred = pred_labels[:len(gt_labels)]
            if page_true:
                correct = sum(t == p for t, p in zip(page_true, page_pred))
                total_correct += correct
                total_samples += len(page_true)
                accuracy = total_correct / total_samples if total_samples else 0
                print(f"Page {self.current_page + 1}: Accuracy {accuracy:.2%}")
                if print_predictions:
                    #print(f"Page {self.current_page + 1}:")
                    for i, (true, pred) in enumerate(zip(page_true, page_pred)):
                        if False: print(f"Block {i}: {pred} - {true}")
            #Train on current page's ground truth by order
            for block, true_label in zip(blocks, gt_labels):
                #add_training_example(block, retrogressive_label_map[true_label])
                add_training_example(block, true_label)
            X, y = get_training_data()
            if X:
                print(f"Training on {len(X)} blocks")
                self.model = train_model(self.model, X, y, epochs=EPOCHS, lr=LEARNING_RATE)
                #Clear training buffer for next page
                training_data.clear()
                normalization_buffer.clear()
            self.current_page += 1
        self.doc.close()
        final_accuracy = total_correct / total_samples if total_samples else 0
        print(f"\nFinal Accuracy: {final_accuracy:.2%}")
        return final_accuracy

    def load_model_weights(self):
        weights_file = "weights_pretrained.pth"
        if os.path.exists(weights_file):
            self.model.load_state_dict(torch.load(weights_file))
            print(f"Loaded pretrained weights")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <basename>")
        sys.exit(1)
    name = sys.argv[1]
    evaluator = PDFEvaluator(f"{name}.pdf", "ground_truth.json")
    final_acc = evaluator.evaluate()
    if False: torch.save(evaluator.model.state_dict(), 'testing_weights.pth')

