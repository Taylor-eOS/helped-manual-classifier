debug = False
epochs = 5
retrain_epochs = 1
learning_rate = 0.0008
print_predictions = False
load_weights_file = False
ground_truth_logging = False
feature_data_file = "feature_data.csv"
dump_features = True
pretrained_file = "pretrained.pt"
pretrained_layout_file = "pretrained_layout.pt"
pretrained_semantic_file = "pretrained_semantic.pt"
pretrain_pdf_path = "shuffled.pdf"
test_file_basename = "test"
pretrain_mask_ratio = 0.1
pretrain_epochs = 25
pretrain_learning_rate = 0.0007
device = "cpu"
print_mistaken_predictions = True
load_pretraining_weights = True
BASE_FEATURES = ['odd_even', 'x0', 'y0', 'width', 'height', 'position', 'letter_count', 'font_size', 'relative_font_size', 'num_lines', 'punctuation_proportion', 'average_words_per_sentence', 'starts_with_number', 'capitalization_proportion', 'average_word_commonality', 'squared_entropy', 'dist_prev_norm', 'dist_next_norm']
SCALES = {
    'x0': 'doc_width',
    'y0': 'doc_height',
    'width': 'doc_width',
    'height': 'doc_height',
    'letter_count': 100,
    'font_size': 24,
    'num_lines': 10,
    'average_words_per_sentence': 10,
    'dist_prev_norm': 1.0,
    'dist_next_norm': 1.0,
    'dist_prev_norm': 0.1,
    'dist_next_norm': 0.1
}
global_features = 4
embedding_components = 5
input_feature_length = len(BASE_FEATURES) + global_features + embedding_components
truncate_embedding_input = 100
debug_get_global_features = False
debug_input_shape = False
debug_model_weight_usage = False
training_examples_per_cycle = 10
max_replay_rounds = 8
use_jina = True

