class Attention(nn.Module):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.units = units
        self.W = nn.Linear(units, units)
        self.U = nn.Linear(units, units)
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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention_units, num_classes=1):
        super(GRUModel, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(attention_units)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru1(x)
        x = self.attention(x)
        x, _ = self.gru2(x.unsqueeze(1))
        x = self.dropout(x[:, -1, :])  # Take the last time-step for classification
        x = self.fc(x)
        return self.sigmoid(x)
    

# Model initialization
vocab_size = len(word_index) + 2
hidden_dim = 16  # Adjust this based on your script
attention_units = 16*2
model = GRUModel(vocab_size, embedding_dim, hidden_dim, attention_units)
model.to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_accuracy = 0
epochs = 100

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