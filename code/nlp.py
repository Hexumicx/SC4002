import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from datasets import load_dataset
import nltk


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Train what?
# 0 for All
# 1 for Dense Layer
# 2 for CNN
# 6 for Attention + multihead + Label Smmoothing + dropout
# 7 for add on to BiGRU
Flag = 10


# Part 1. Preparing Word Embeddings
# (a) What is the size of the vocabulary formed from your training data?
with open("pytorch_result.txt", "w") as file:
    print("Part 1. Preparing Word Embeddings:", file=file)

dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset ['train']
validation_dataset = dataset ['validation']
test_dataset = dataset ['test']

nltk.download('punkt')
vocab = set()
for text in train_dataset['text']:
    ls = nltk.word_tokenize(text)
    for word in ls:
        if word.isalpha(): vocab.add(word)

print("Size of the vocabulary:", len(vocab))

with open("pytorch_result.txt", "a") as file:
    print("1(a) Size of the vocabulary:", len(vocab), file=file)

# (b) We use OOV (out-of-vocabulary) to refer to those words appeared in the training data but not in the Word2vec (or Glove) dictionary. How many OOV words exist in your training data?

import gensim.downloader as api
for key in api.info()['models'].keys():
    print(key)

# embedding_model = api.load("glove-wiki-gigaword-100")
embedding_model = api.load("glove-wiki-gigaword-100")

oov_words = set()
for word in vocab:
    if word not in embedding_model:
        oov_words.add(word)

print("Number of OOV words:", len(oov_words))

# with open("/app/result/result.txt", "a") as file:
#     print("1(b) Number of OOV words:", len(oov_words), file=file)

# (c) The existence of the OOV words is one of the well-known limitations of Word2vec (or Glove). Without using any transformer-based language models (e.g., BERT, GPT, T5), what do you think is the best strategy to mitigate such limitation? Implement your solution in your source code. Show the corresponding code snippet.

def wordtovec(word):
    if word in embedding_model:
        return embedding_model[word]
    else:
        return np.zeros(embedding_model.vector_size)
dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset ['train']
validation_dataset = dataset ['validation']
test_dataset = dataset ['test']

# Part 2. Model Training & Evaluation - RNN
vocab_size = len(embedding_model.index_to_key) + 2
embedding_dim = embedding_model.vector_size
word_index = {word: index+2 for index, word in enumerate(embedding_model.index_to_key)} # index 0 is reserved for padding
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in word_index.items():
    if word in embedding_model:
        embedding_matrix[idx] = embedding_model[word]

def tokenize(text, word_index):
    ls = nltk.word_tokenize(text)
    return [word_index[word] if word in word_index else 1 for word in ls]


X_train = [torch.tensor(tokenize(text, word_index), dtype=torch.long) for text in train_dataset['text']]
X_val = [torch.tensor(tokenize(text, word_index), dtype=torch.long) for text in validation_dataset['text']]
X_test = [torch.tensor(tokenize(text, word_index), dtype=torch.long) for text in test_dataset['text']]

max_length = max(len(seq) for seq in X_train)

X_train = pad_sequence([seq[:max_length] for seq in X_train], batch_first=True, padding_value=0)
X_val = pad_sequence([seq[:max_length] for seq in X_val], batch_first=True, padding_value=0)
X_test = pad_sequence([seq[:max_length] for seq in X_test], batch_first=True, padding_value=0)

y_train = torch.tensor(train_dataset['label'], dtype=torch.float32)
y_val = torch.tensor(validation_dataset['label'], dtype=torch.float32)
y_test = torch.tensor(test_dataset['label'], dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class Attention(nn.Module):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.units = units
        self.W = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, hidden_states):
        # score = torch.tanh(self.W(hidden_states))
        # Assuming hidden_states is of shape (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = hidden_states.size()

        if hidden_dim != self.units:
            raise ValueError(f"Expected hidden_dim to be {self.units}, but got {hidden_dim}")

        # Apply the linear layer to get scores
        score = torch.tanh(self.W(hidden_states))  # shape: (batch_size, seq_len, units)
        
        # Compute attention weights
        attention_weights = torch.softmax(self.V(score), dim=1)  # shape: (batch_size, seq_len, 1)

        # Perform element-wise multiplication and compute the context vector
        context_vector = attention_weights * hidden_states  # Broadcasting works automatically here
        context_vector = context_vector.sum(dim=1)  # Sum along the sequence dimension

        return context_vector


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, attention_units, dropout_rate, num_classes=1):
        super(GRUModel, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.attention = Attention(attention_units)
        self.gru2 = nn.GRU(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim2 * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru1(x)
        x = self.attention(x)
        x, _ = self.gru2(x.unsqueeze(1))
        x = self.dropout(x[:, -1, :])  # Take the last time-step for classification
        x = self.fc(x)
        return self.sigmoid(x)
    
def biGRU_attention_biGRU(hd1, hd2, dr, epochs):
    # Model initialization
    vocab_size = len(word_index) + 2
    hidden_dim1 = hd1  # Adjust this based on your script
    hidden_dim2 = hd2
    attention_units = hidden_dim1*2
    dropout_rate = dr
    model = GRUModel(vocab_size, embedding_dim, hidden_dim1, hidden_dim2, attention_units, dropout_rate)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train, total_train = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = (outputs.squeeze() > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_accuracy = correct_train / total_train
        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Track best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Saved new best model with accuracy: {best_accuracy:.4f}")

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

    with open("pytorch_result.txt", "a") as file:
        print(f"Final Train Accuracy: {train_accuracy:.4f}", f"Best Validation Accuracy: {best_accuracy:.4f}", f"Test Accuracy: {test_accuracy:.4f}", file=file)

with open("pytorch_result.txt", "a") as file:
    print("hd1 = 32, hd2 = 16, dr = 0.5, epochs = 100", file=file)
biGRU_attention_biGRU(32, 16, 0.5, 100)