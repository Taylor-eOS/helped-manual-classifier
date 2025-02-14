import os
import argparse
import fitz
from model_util import predict_blocks, BlockClassifier
from main_script import ManualClassifierGUI

def load_ground_truth(gt_file_path):
    gt_blocks = []
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    current_block_lines = []
    for line in lines:
        if line.strip() == "":
            if current_block_lines:
                #The last line should be the page indicator (e.g. <1>)
                block_text = "\n".join(current_block_lines[:-1])
                page_line = current_block_lines[-1].strip()
                try:
                    page = int(page_line.strip("<>"))
                except ValueError:
                    page = None
                # Extract the inner text of the tag to use as the true label.
                if block_text.startswith("<") and ">" in block_text:
                    tag_end = block_text.find(">")
                    tag_start = block_text.rfind("</")
                    if tag_start != -1:
                        inner = block_text[tag_end+1:tag_start]
                        inner_lines = inner.splitlines()
                        true_label = inner_lines[0].strip() if inner_lines else "Unknown"
                    else:
                        true_label = "Unknown"
                    key_text = true_label  # Use the inner text for key generation.
                else:
                    true_label = "Unknown"
                    key_text = block_text.splitlines()[0] if block_text else ""
                key = f"{page}_{key_text[:30]}"
                gt_blocks.append({
                    'page': page,
                    'text': block_text,
                    'true_label': true_label,
                    'key': key
                })
            current_block_lines = []
        else:
            current_block_lines.append(line)
    return gt_blocks

def create_ground_truth_dict(gt_blocks):
    gt_dict = {}
    for block in gt_blocks:
        gt_dict[block['key']] = block['true_label']
    return gt_dict

#Returns a key for matching a block based on its page number and the first 30 characters of its text.
def get_block_key(block):
    text = block.get('text', "")
    first_line = text.splitlines()[0] if text else ""
    key = f"{block.get('page', 'None')}_{first_line[:30]}"
    return key

#Subclass the ManualClassifierGUI to make a headless tester.
class TestClassifier(ManualClassifierGUI):
    def __init__(self, pdf_path):
        #Initialize most of the attributes but skip the GUI startup.
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = self.doc.page_count
        self.all_blocks = [None] * self.total_pages
        self.block_classifications = []
        self.current_page = 0
        self.page_buffer = []
        self.current_page_blocks = []
        self.global_indices = []
        self.mlp_model = BlockClassifier()
        #No need for a processing lock or GUI elements during testing.
        self.zoom = 2
        self.scale = 1.0

    #Extract blocks from all pages and store them.
    def load_all_pages(self):
        for page_num in range(self.total_pages):
            blocks = self.extract_page_geometric_features(page_num)
            #Ensure each block has a page number and its text.
            for block in blocks:
                block['page'] = page_num
            self.all_blocks[page_num] = blocks
            self.block_classifications.extend(['0'] * len(blocks))
        #Flatten all blocks into a single list.
        all_blocks = []
        for page_blocks in self.all_blocks:
            all_blocks.extend(page_blocks)
        return all_blocks

def main():
    parser = argparse.ArgumentParser(description="Test PDF Block Classification")
    parser.add_argument("file_name", help="Path to the file(s) without file ending")
    args = parser.parse_args()
    gt_path = os.path.abspath(f"{args.file_name}.txt")
    pdf_path = os.path.abspath(f"{args.file_name}.pdf")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    gt_blocks = load_ground_truth(gt_path)
    gt_dict = create_ground_truth_dict(gt_blocks)
    #print("Ground truth dictionary:", gt_dict)
    tester = TestClassifier(pdf_path)
    all_blocks = tester.load_all_pages()
    #Run predictions on all extracted blocks using your model.
    predicted_labels = predict_blocks(tester.mlp_model, all_blocks)
    #Compare predictions with ground truth.
    total, correct = 0, 0
    mismatches = []
    for block, pred_label in zip(all_blocks, predicted_labels):
        key = get_block_key(block)
        total += 1
        #If the key is not found in the ground truth, assume the block should be excluded.
        true_label = gt_dict.get(key, "Exclude")
        if pred_label == true_label:
            correct += 1
        else:
            mismatches.append({
                'page': block.get('page'),
                'key': key,
                'true_label': true_label,
                'pred_label': pred_label,
                'text_snippet': block.get('text', "")[:50]})
    accuracy = (correct / total * 100) if total else 0
    print(f"Total blocks processed: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%\n")
    if mismatches:
        for mismatch in mismatches:
            accuracy = (correct / total * 100) if total else 0
            with open("test_results.txt", "w", encoding="utf-8") as f:
                f.write(f"Total blocks processed: {total}\n")
                f.write(f"Correct predictions: {correct}\n")
                f.write(f"Accuracy: {accuracy:.2f}%\n\n")
                if mismatches:
                    f.write("Mismatches:\n")
                    for mismatch in mismatches:
                        f.write(f"Page {mismatch['page']} | Key: {mismatch['key']} | True: {mismatch['true_label']} / Pred: {mismatch['pred_label']} | Snippet: {mismatch['text_snippet']}\n")

def get_block_key(block):
    text = block.get('text', "")
    first_line = text.splitlines()[0] if text else ""
    page = block.get('page', 'None')
    if isinstance(page, int):
        page = page + 1
    key = f"{page}_{first_line[:30]}"
    print(f"DEBUG: Extracted PDF block key: page={page}, key='{key}', snippet='{first_line}'")
    return key

def load_ground_truth_debug(gt_file_path):
    gt_blocks = []
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    current_block_lines = []
    for line in lines:
        if line.strip() == "":
            if current_block_lines:
                block_text = "\n".join(current_block_lines[:-1])
                page_line = current_block_lines[-1].strip()
                try:
                    page = int(page_line.strip("<>"))
                except ValueError:
                    page = None
                if block_text.startswith("<") and ">" in block_text:
                    tag_end = block_text.find(">")
                    tag_start = block_text.rfind("</")
                    if tag_start != -1:
                        inner = block_text[tag_end+1:tag_start]
                        inner_lines = inner.splitlines()
                        true_label = inner_lines[0].strip() if inner_lines else "Unknown"
                    else:
                        true_label = "Unknown"
                else:
                    true_label = "Unknown"
                # Use the extracted true_label for the key instead of the raw block_text.
                key_text = true_label if true_label != "Unknown" else (block_text.splitlines()[0] if block_text else "")
                key = f"{page}_{key_text[:30]}"
                print(f"DEBUG: Loaded GT block: page={page}, key='{key}', true_label='{true_label}'")
                gt_blocks.append({
                    'page': page,
                    'text': block_text,
                    'true_label': true_label,
                    'key': key
                })
            current_block_lines = []
        else:
            current_block_lines.append(line)
    return gt_blocks

if __name__ == "__main__":
    main()
