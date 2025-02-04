import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Mapping between labels and integers.
LABEL_MAP = {"Header": 0, "Body": 1, "Footer": 2, "Quote": 3, "Exclude": 4}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# A fast, agile model for quick adaptation.
class BlockClassifier(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, output_dim=5):
        """
        A simple one-hidden-layer network with 64 neurons.
        This model is small enough to adapt quickly from only a few examples.
        """
        super(BlockClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def block_to_features(block):
    # Convert a block dictionary into a fixed-length feature vector.
    features = [
        block.get("x0", 0),
        block.get("y0", 0),
        block.get("x1", 0),
        block.get("y1", 0),
        block.get("height", 0),
        block.get("width", 0),
        block.get("position", 0),
        block.get("letter_count", 0),
        block.get("font_size", 0),
        block.get("relative_font_size", 0),
        block.get("num_lines", 0),
        block.get("punctuation_proportion", 0),
        block.get("average_words_per_sentence", 0),
        block.get("starts_with_number", 0),
        block.get("capitalization_proportion", 0),
        block.get("average_word_commonality", 0),
        block.get("squared_entropy", 0)
    ]
    return features

def train_model(model, train_features, train_labels, epochs=20, lr=0.01):
    """
    Train the model with a high learning rate and few epochs for fast adaptation.
    For long-term intelligence, you might consider periodically integrating
    more data into a larger model offline.
    """
    if len(train_features) == 0:
        return model  # Nothing to train on.
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

def predict_blocks(model, blocks):
    model.eval()
    features = [block_to_features(block) for block in blocks]
    X = torch.tensor(np.array(features), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
    pred_labels = [INV_LABEL_MAP[int(p)] for p in preds]
    return pred_labels

def get_training_data(blocks):
    """
    From a list of blocks (typically on one page), extract training examples.
    Blocks that have been manually labeled (i.e. block["type"] != '0') are used.
    """
    train_features = []
    train_labels = []
    for block in blocks:
        if block.get("type", '0') != '0':
            label_str = block["type"]
            if label_str in LABEL_MAP:
                train_features.append(block_to_features(block))
                train_labels.append(LABEL_MAP[label_str])
    return np.array(train_features), np.array(train_labels)

# Example usage:
if __name__ == "__main__":
    # Instantiate the agile model.
    model = AgileBlockClassifier()

    # Suppose these are blocks with manual labels from a user.
    blocks = [
        {"x0": 10, "y0": 10, "x1": 100, "y1": 50, "height": 40, "width": 90,
         "position": 1, "letter_count": 30, "font_size": 12, "relative_font_size": 1.0,
         "num_lines": 1, "punctuation_proportion": 0.1, "average_words_per_sentence": 5,
         "starts_with_number": 0, "capitalization_proportion": 0.5, "average_word_commonality": 0.8,
         "squared_entropy": 0.2, "type": "Header"},
        {"x0": 15, "y0": 60, "x1": 110, "y1": 120, "height": 60, "width": 95,
         "position": 2, "letter_count": 150, "font_size": 10, "relative_font_size": 0.8,
         "num_lines": 3, "punctuation_proportion": 0.2, "average_words_per_sentence": 7,
         "starts_with_number": 0, "capitalization_proportion": 0.3, "average_word_commonality": 0.7,
         "squared_entropy": 0.3, "type": "Body"}
    ]

    # Extract training data from blocks.
    train_features, train_labels = get_training_data(blocks)

    # Quickly fine-tune the model on these examples.
    model = train_model(model, train_features, train_labels)

    # Use the model to predict labels for new blocks.
    predictions = predict_blocks(model, blocks)
    print("Predictions:", predictions)

