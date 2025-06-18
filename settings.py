debug = False
epochs = 5
learning_rate = 0.0008
print_predictions = False
load_weights_file = False
ground_truth_logging = False
feature_data_file = "feature_data.csv"
print_features = False
pretrained_file = "pretrained.pt"
pretrain_pdf_path = "shuffled.pdf"
pretrain_mask_ratio = 0.1
pretrain_epochs = 25
pretrain_learning_rate = 0.0007
device = "cpu"
print_mistaken_predictions = True
save_testing_weights = False
load_pretraining_weights = False
BASE_FEATURES = [
    'odd_even', 'x0', 'y0', 'width', 'height', 'position',
    'letter_count', 'font_size', 'relative_font_size',
    'num_lines', 'punctuation_proportion', 'average_words_per_sentence',
    'starts_with_number', 'capitalization_proportion',
    'average_word_commonality', 'squared_entropy'
]
SCALES = {
    'x0': 'doc_width',
    'y0': 'doc_height',
    'width': 'doc_width',
    'height': 'doc_height',
    'letter_count': 100,
    'font_size': 24,
    'num_lines': 10,
    'average_words_per_sentence': 10
}

