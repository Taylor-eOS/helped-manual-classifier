import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class BlockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

training_data = deque(maxlen=100)

def get_training_data():
    if not training_data:
        return [], []
    features, labels = zip(*training_data)
    return list(features), list(labels)

def add_training_example(block, label):
    label_map = {'Header': 0, 'Body': 1, 'Footer': 2, 'Quote': 3, 'Exclude': 4}
    features = get_features(block)
    training_data.append((features, label_map[label]))

def train_model(model, features, labels, epochs=5, lr=0.05):
    if not features:
        return model
    X_train = torch.tensor(features, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model

def predict_blocks(model, blocks):
    if not blocks:
        return []
    model.eval()
    X_test = torch.tensor([get_features(b) for b in blocks], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
    return [['Header', 'Body', 'Footer', 'Quote', 'Exclude'][p] for p in predictions.tolist()]

def get_features(block):
    return [
        block['x0'], block['y0'],
        block['width']/612, block['height']/792,
        block['position'],
        block['letter_count']/100,
        block['font_size']/24,
        block['relative_font_size'],
        block['num_lines']/10,
        block['punctuation_proportion'],
        block['average_words_per_sentence']/10,
        block['starts_with_number'],
        block['capitalization_proportion'],
        block['average_word_commonality'],
        block['squared_entropy']
    ]
