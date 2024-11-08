# Part 0. Data Prepraration
from datasets import load_dataset
dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset ['train']
validation_dataset = dataset ['validation']
test_dataset = dataset ['test']

Flag = 3

# Part 1. Preparing Word Embeddings
# (a) What is the size of the vocabulary formed from your training data?
with open("/app/result/result.txt", "w") as file:
    print("Part 1. Preparing Word Embeddings:", file=file)

import nltk
import numpy as np
nltk.download('punkt')
# nltk.download('punkt_tab')

vocab = set()
for text in train_dataset['text']:
    ls = nltk.word_tokenize(text)
    for word in ls:
        if word.isalpha(): vocab.add(word)

print("Size of the vocabulary:", len(vocab))

with open("/app/result/result.txt", "a") as file:
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

with open("/app/result/result.txt", "a") as file:
    print("1(b) Number of OOV words:", len(oov_words), file=file)

# (c) The existence of the OOV words is one of the well-known limitations of Word2vec (or Glove). Without using any transformer-based language models (e.g., BERT, GPT, T5), what do you think is the best strategy to mitigate such limitation? Implement your solution in your source code. Show the corresponding code snippet.

def wordtovec(word):
    if word in embedding_model:
        return embedding_model[word]
    else:
        return np.zeros(embedding_model.vector_size)
    
# Part 2. Model Training & Evaluation - RNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import random

vocab_size = len(embedding_model.index_to_key) + 1
embedding_dim = embedding_model.vector_size
print("embedding_dim : ", embedding_dim)
word_index = {word: index+1 for index, word in enumerate(embedding_model.index_to_key)} # index 0 is reserved for padding
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in word_index.items():
    if word in embedding_model:
        embedding_matrix[idx] = embedding_model[word]

def tokenize(text, word_index):
    ls = nltk.word_tokenize(text)
    return [word_index[word] for word in ls if word in word_index]

X_train = [tokenize(text, word_index) for text in train_dataset['text']]
X_val = [tokenize(text, word_index) for text in validation_dataset['text']]
X_test = [tokenize(text, word_index) for text in test_dataset['text']]
max_length = max(len(seq) for seq in X_train)

X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

y_train = np.array(train_dataset['label'])
y_val = np.array(validation_dataset['label'])
y_test = np.array(test_dataset['label'])

# Model Training - Grid Search

from tensorflow.keras.callbacks import Callback
best_accuracy = {}
class CustomCallback(Callback):
    accuracy = 0
    cur_key = ""
    epochs = 0
    optimizer = ""
    batch_size = 0
    lr = 0
    def on_train_begin(self, logs=None):
        self.accuracy = 0

    def on_train_end(self, logs=None):
        global best_accuracy
        if self.accuracy > best_accuracy.get("accuracy", 0):
            best_accuracy = {
                "accuracy": self.accuracy,
                "epoch": self.epochs,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "lr": self.lr
            }
            print("Saved best accuracy for current run:", self.accuracy, "at epoch", self.epochs)
            self.model.save(filepath="best_model.keras")
        print("Run completed on:")
        print(self.cur_key)
        print("Best accuracy for current run:", self.accuracy, "at epoch", self.epochs)
        print("Training ended")
    
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs['val_accuracy']
        if val_accuracy > self.accuracy:
            self.accuracy = val_accuracy
            self.epochs = epoch

    def set_key(self, optimizer, batch_size, lr):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.cur_key = f"optimizer: {optimizer}, batch_size: {batch_size}, lr: {lr}"

def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=False),  # Embedding layer is frozen
        SimpleRNN(16, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, early_stopping]
    )
    return model, history

if not Flag:
    for batch_size in [16, 32, 64, 128]:
        for lr in [0.005, 0.01, 0.05, 0.1]:
            for optimizer in ['adam', 'sgd', 'rmsprop', 'adagrad']:
                train_model(optimizer, 100, batch_size, lr)

    print("Best accuracy: ", best_accuracy)
    with open("/app/result/result.txt", "a") as file:
        print("2(a) Best Accuracy for training:", best_accuracy, file=file)

    #Train the best accuracy model
    model, history = train_model("adagrad", 40, 64, 0.01)

    #Run it on test set
    best_model = tf.keras.models.load_model("best_model.keras")
    accuracy = best_model.evaluate(X_val, y_val)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("2(b) Result on test Set:", accuracy[1], file=file)

    #Whats this for?
    best_model.get_compile_config()

#-----------------------Mean Pooling ----------------------------
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling1D
def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_mean.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=False),  # Embedding layer is frozen
        SimpleRNN(16, return_sequences=True),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag:
    model, history = train_model("adagrad", 100, 64, 0.01)

    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy for training (Mean Pooling):", best_accuracy, file=file)

    best_model = tf.keras.models.load_model("model_mean.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy on test Set (Mean Pooling):", accuracy[1], file=file)

#-----------------------Max Pooling ----------------------------
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D
def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_max.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=False),  # Embedding layer is frozen
        SimpleRNN(16, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping]
    )
    return model, history
if not Flag:
    model, history = train_model("adagrad", 100, 64, 0.01)
    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy for training (Max Pooling):", best_accuracy, file=file)

    best_model = tf.keras.models.load_model("model_max.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy on test Set (Max Pooling):", accuracy[1], file=file)
#-----------------------Dense Layer ----------------------------
from tensorflow.keras.layers import Flatten
def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_dense.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=False),  # Embedding layer is frozen
        SimpleRNN(16, return_sequences=True),
        Flatten(),
        Dense(62, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag or Flag == 1:
    model, history = train_model("adagrad", 100, 64, 0.01)
    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy for training (Dense Layer):", best_accuracy, file=file)

    best_model = tf.keras.models.load_model("model_dense.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy on test Set (Dense Layer):", accuracy[1], file=file)

#Part 3 Enhancement

# 1. Instead of keeping the word embeddings fixed, now update the word embeddings (the same way as model parameters) during the training process.
# tf.random.set_seed(0)
# np.random.seed(0)
# random.seed(0)
# model = Sequential([
#     Embedding(input_dim=vocab_size,
#               output_dim=embedding_dim,
#               weights=[embedding_matrix],
#               trainable=True),
#     SimpleRNN(16, return_sequences=False),
#     Dense(1, activation='sigmoid')
# ])
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) #Static learning rate
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=20,
#     batch_size=32,
#     callbacks=[early_stopping]
# )

# 2. As discussed in Question 1(c), apply your solution in mitigating the influence of OOV words and train your model again.
vocab_size = len(embedding_model.index_to_key) + 2 # 0 is reserved for padding, 1 is reserved for OOV
embedding_dim = embedding_model.vector_size
word_index = {word: index+2 for index, word in enumerate(embedding_model.index_to_key)} # index 0 is reserved for padding
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in word_index.items():
    if word in embedding_model:
        embedding_matrix[idx] = embedding_model[word]

def tokenize(text, word_index):
    ls = nltk.word_tokenize(text)
    return [word_index[word] if word in word_index else 1 for word in ls]

X_train = [tokenize(text, word_index) for text in train_dataset['text']]
X_val = [tokenize(text, word_index) for text in validation_dataset['text']]
X_test = [tokenize(text, word_index) for text in test_dataset['text']]
max_length = max(len(seq) for seq in X_train)

X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

y_train = np.array(train_dataset['label'])
y_val = np.array(validation_dataset['label'])
y_test = np.array(test_dataset['label'])

def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_oov.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=False),  # Embedding layer is frozen
        SimpleRNN(16, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag:
    train_model("adagrad", 100, 64, 0.01)
    with open("/app/result/result.txt", "a") as file:
        print("3.2 Best Accuracy for training (Update word embedings and mitigate OOV words): ", best_accuracy, file=file)

    best_model = tf.keras.models.load_model("model_oov.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.2 Best Accuracy on test Set (Update word embedings and mitigate OOV words): ", accuracy[1], file=file)

#3. Keeping the above two adjustments, replace your simple RNN model in Part 2 with a biLSTM model and a biGRU model, incorporating recurrent computations in both directions and stacking multiple layers if possible
from tensorflow.keras.layers import Bidirectional, LSTM


#----------------------- BiLSTM ----------------------------
def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        Bidirectional(LSTM(16, return_sequences=False)),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag:
    train_model("adagrad", 100, 64, 0.01)
    with open("/app/result/result.txt", "a") as file:
        print("3.3 Best Accuracy for training (Bidirectional LSTM): ", best_accuracy, file=file)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.3 Best Accuracy on test Set (Bidirectional LSTM): ", accuracy[1], file=file)

#----------------------- BiGRU ----------------------------
from tensorflow.keras.layers import Bidirectional, GRU
def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        Bidirectional(GRU(16, return_sequences=False)),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag:
    train_model("adagrad", 100, 64, 0.01)
    with open("/app/result/result.txt", "a") as file:
        print("3.3 Best Accuracy for training (Bidirectional GRU): ", best_accuracy, file=file)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.3 Best Accuracy on test Set (Bidirectional GRU): ", accuracy[1], file=file)

# 4. Keeping the above two adjustments, replace your simple RNN model in Part 2 with a Convolutional Neural Network (CNN) to produce sentence representations and perform sentiment classification
from tensorflow.keras.layers import Convolution1D, Flatten

def train_model(optimizer, epochs, batch_size, lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is trainable
        Convolution1D(16, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag or Flag == 1:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.4 Best Accuracy for training (CNN): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.4 Best Accuracy on test Set (CNN): ", accuracy[1], file=file)

#5. Further Improvement
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MultiHeadAttention, Input, LayerNormalization, Add
from tensorflow.keras.layers import Reshape
from tensorflow.keras import regularizers

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

epoch = 100

with open("/app/result/result.txt", "a") as file:
    print("--------3.5 Enhancement--------", file=file)


# Add Attention Layer

@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_dim)
        score = tf.nn.tanh(self.W(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config
    
# 11            
def train_model_all(optimizer, epochs, batch_size, lr, dim1, type = 'RNN', dim2 = None, bi= None, dimA = None, dense1Dim = None, dense2Dim = None, dropout = None, l2 = None):
    tf.keras.backend.clear_session()

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True), 
    ])

    if l2:
        #RNN
        if type == 'RNN':
            if (bi and dimA) or (bi and dim2):
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=True, kernel_regularizer=regularizers.l2(l2))))
                if dropout:
                    model.add(Dropout(dropout))
            elif bi and not dimA:
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=False, kernel_regularizer=regularizers.l2(l2))))
                if dropout:
                    model.add(Dropout(dropout))
            elif (not bi and dimA) or (not bi and dim2):
                model.add(SimpleRNN(dim1, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
                if dropout:
                    model.add(Dropout(dropout))
            else:
                model.add(SimpleRNN(dim1, return_sequences=False, kernel_regularizer=regularizers.l2(l2)))
                if dropout:
                    model.add(Dropout(dropout))


            if dim2:
                if (bi and dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=True, kernel_regularizer=regularizers.l2(l2))))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (bi and not dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=False, kernel_regularizer=regularizers.l2(l2))))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (not bi and dimA):
                    model.add(SimpleRNN(dim2, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
                    if dropout:
                        model.add(Dropout(dropout))
                else:
                    model.add(SimpleRNN(dim2, return_sequences=False, kernel_regularizer=regularizers.l2(l2)))
                    if dropout:
                        model.add(Dropout(dropout))

        #LSTM
        elif type == 'LSTM':
            if (bi and dimA) or (bi and dim2):
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=True, kernel_regularizer=regularizers.l2(l2))))
                if dropout:
                    model.add(Dropout(dropout))
            elif bi and not dimA:
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=False, kernel_regularizer=regularizers.l2(l2))))
                if dropout:
                    model.add(Dropout(dropout))
            elif (not bi and dimA) or (not bi and dim2):
                model.add(SimpleRNN(dim1, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
                if dropout:
                    model.add(Dropout(dropout))
            else:
                model.add(SimpleRNN(dim1, return_sequences=False, kernel_regularizer=regularizers.l2(l2)))
                if dropout:
                    model.add(Dropout(dropout))


            if dim2:
                if (bi and dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=True, kernel_regularizer=regularizers.l2(l2))))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (bi and not dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=False, kernel_regularizer=regularizers.l2(l2))))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (not bi and dimA):
                    model.add(SimpleRNN(dim2, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
                    if dropout:
                        model.add(Dropout(dropout))
                else:
                    model.add(SimpleRNN(dim2, return_sequences=False, kernel_regularizer=regularizers.l2(l2)))
                    if dropout:
                        model.add(Dropout(dropout))
        
        #GRU
        elif type == 'GRU':
            if (bi and dimA) or (bi and dim2):
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=True, kernel_regularizer=regularizers.l2(l2))))
                if dropout:
                    model.add(Dropout(dropout))
            elif bi and not dimA:
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=False, kernel_regularizer=regularizers.l2(l2))))
                if dropout:
                    model.add(Dropout(dropout))
            elif (not bi and dimA) or (not bi and dim2):
                model.add(SimpleRNN(dim1, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
                if dropout:
                    model.add(Dropout(dropout))
            else:
                model.add(SimpleRNN(dim1, return_sequences=False, kernel_regularizer=regularizers.l2(l2)))
                if dropout:
                    model.add(Dropout(dropout))


            if dim2:
                if (bi and dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=True, kernel_regularizer=regularizers.l2(l2))))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (bi and not dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=False, kernel_regularizer=regularizers.l2(l2))))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (not bi and dimA):
                    model.add(SimpleRNN(dim2, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
                    if dropout:
                        model.add(Dropout(dropout))
                else:
                    model.add(SimpleRNN(dim2, return_sequences=False, kernel_regularizer=regularizers.l2(l2)))
                    if dropout:
                        model.add(Dropout(dropout))

    else:
        #RNN
        if type == 'RNN':
            if (bi and dimA) or (bi and dim2):
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=True)))
                if dropout:
                    model.add(Dropout(dropout))
            elif bi and not dimA:
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=False)))
                if dropout:
                    model.add(Dropout(dropout))
            elif (not bi and dimA) or (not bi and dim2):
                model.add(SimpleRNN(dim1, return_sequences=True))
                if dropout:
                    model.add(Dropout(dropout))
            else:
                model.add(SimpleRNN(dim1, return_sequences=False))
                if dropout:
                    model.add(Dropout(dropout))


            if dim2:
                if (bi and dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=True)))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (bi and not dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=False)))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (not bi and dimA):
                    model.add(SimpleRNN(dim2, return_sequences=True))
                    if dropout:
                        model.add(Dropout(dropout))
                else:
                    model.add(SimpleRNN(dim2, return_sequences=False))
                    if dropout:
                        model.add(Dropout(dropout))

        #LSTM
        elif type == 'LSTM':
            if (bi and dimA) or (bi and dim2):
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=True)))
                if dropout:
                    model.add(Dropout(dropout))
            elif bi and not dimA:
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=False)))
                if dropout:
                    model.add(Dropout(dropout))
            elif (not bi and dimA) or (not bi and dim2):
                model.add(SimpleRNN(dim1, return_sequences=True))
                if dropout:
                    model.add(Dropout(dropout))
            else:
                model.add(SimpleRNN(dim1, return_sequences=False))
                if dropout:
                    model.add(Dropout(dropout))


            if dim2:
                if (bi and dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=True)))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (bi and not dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=False)))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (not bi and dimA):
                    model.add(SimpleRNN(dim2, return_sequences=True))
                    if dropout:
                        model.add(Dropout(dropout))
                else:
                    model.add(SimpleRNN(dim2, return_sequences=False))
                    if dropout:
                        model.add(Dropout(dropout))
        
        #GRU
        elif type == 'GRU':
            if (bi and dimA) or (bi and dim2):
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=True)))
                if dropout:
                    model.add(Dropout(dropout))
            elif bi and not dimA:
                model.add(Bidirectional(SimpleRNN(dim1, return_sequences=False)))
                if dropout:
                    model.add(Dropout(dropout))
            elif (not bi and dimA) or (not bi and dim2):
                model.add(SimpleRNN(dim1, return_sequences=True))
                if dropout:
                    model.add(Dropout(dropout))
            else:
                model.add(SimpleRNN(dim1, return_sequences=False))
                if dropout:
                    model.add(Dropout(dropout))


            if dim2:
                if (bi and dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=True)))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (bi and not dimA):
                    model.add(Bidirectional(SimpleRNN(dim2, return_sequences=False)))
                    if dropout:
                        model.add(Dropout(dropout))
                elif (not bi and dimA):
                    model.add(SimpleRNN(dim2, return_sequences=True))
                    if dropout:
                        model.add(Dropout(dropout))
                else:
                    model.add(SimpleRNN(dim2, return_sequences=False))
                    if dropout:
                        model.add(Dropout(dropout))

    if dimA:
        model.add(Attention(units=dimA))
        if dropout:
            model.add(Dropout(dropout))

    if dense1Dim:
        model.add(Dense(dense1Dim, activation='relu'))

    if dense2Dim:
        model.add(Dense(dense2Dim, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    

    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, checkpoint_callback, early_stopping]
    )
    return model, history


# if not Flag or Flag == 2 or Flag == 11:
#     for batch_size in [64]:
#         for lr in [0.01]:
#             for optimizer in ['adagrad']:
#                 for type in ['RNN', 'LSTM', 'GRU']:
#                     for bi in [None, True]:
#                         for dropout in [None, 0.2, 0.5, 0.7]:
#                             for dim1 in [16, 32 , 64 , embedding_dim]:
#                                 for dim2 in [None, 16, 32 , 64 , embedding_dim]:
#                                     for dimA in [None, 16, 32, 64, embedding_dim, 200, 1000]:
#                                         for dense1Dim in [None, 16, 32, 64, 128]:
#                                             for dense2Dim in [None, 16, 32, 64, 128]:
#                                                 for l2 in [None, 0.01, 0.001, 0.0001]:

#                                                     model, history = train_model_all(optimizer, epoch, batch_size, lr, dim1, type = type, dim2 = dim2, bi = bi, dimA = dimA, dense1Dim= dense1Dim, dense2Dim = dense2Dim, dropout = dropout, l2 = l2)

#                                                     best_model = tf.keras.models.load_model("model_combined.keras")
#                                                     accuracy = best_model.evaluate(X_test, y_test)
#                                                     with open("/app/result/result.txt", "a") as file:
#                                                         print(f"3.5 Best Accuracy for training batch_size : {batch_size} , lr: {lr}, optim: {optimizer}, dim1: {dim1}, type: {type}, dim2: {dim2}, bi: {bi}, dimA: {dimA}, dense1Dim: {dense1Dim}, dense2Dim: {dense2Dim}, dropout: {dropout}, l2: {l2} : ", best_accuracy, file=file)
#                                                     print("Test accuracy:", accuracy[1])
#                                                     with open("/app/result/result.txt", "a") as file:
#                                                         print(f"3.5 Best Accuracy for testing batch_size : {batch_size} , lr: {lr}, optim: {optimizer}, dim1: {dim1}, type: {type}, dim2: {dim2}, bi: {bi}, dimA: {dimA}, dense1Dim: {dense1Dim}, dense2Dim: {dense2Dim}, dropout: {dropout}, l2: {l2} : ", accuracy[1], file=file)



if not Flag or Flag == 2 or Flag == 11:
    for batch_size in [64]:
        for lr in [0.01]:
            for optimizer in ['adagrad']:
                for type in ['GRU']:
                    for bi in [True]:
                        for dropout in [0.7]:
                            for dim1 in [64 , embedding_dim]:
                                for dim2 in [None]:
                                    for dimA in [16, 32, 64, embedding_dim, 1000]:
                                        for dense1Dim in [None, 16, 32]:
                                            if dense1Dim:
                                                for dense2Dim in [None, 16]:
                                                    for l2 in [None, 0.001]:

                                                        model, history = train_model_all(optimizer, epoch, batch_size, lr, dim1, type = type, dim2 = dim2, bi = bi, dimA = dimA, dense1Dim= dense1Dim, dense2Dim = dense2Dim, dropout = dropout, l2 = l2)

                                                        best_model = tf.keras.models.load_model("model_combined.keras")
                                                        accuracy = best_model.evaluate(X_test, y_test)
                                                        with open("/app/result/result.txt", "a") as file:
                                                            print(f"3.5 Best Accuracy for training batch_size : {batch_size} , lr: {lr}, optim: {optimizer}, dim1: {dim1}, type: {type}, dim2: {dim2}, bi: {bi}, dimA: {dimA}, dense1Dim: {dense1Dim}, dense2Dim: {dense2Dim}, dropout: {dropout}, l2: {l2} : ", best_accuracy, file=file)
                                                        print("Test accuracy:", accuracy[1])
                                                        with open("/app/result/result.txt", "a") as file:
                                                            print(f"3.5 Best Accuracy for testing batch_size : {batch_size} , lr: {lr}, optim: {optimizer}, dim1: {dim1}, type: {type}, dim2: {dim2}, bi: {bi}, dimA: {dimA}, dense1Dim: {dense1Dim}, dense2Dim: {dense2Dim}, dropout: {dropout}, l2: {l2} : ", accuracy[1], file=file)



# actual

#ADJUST PARAM ON GOOD ON CAN REACH 78% without early stopping and no schedule learning rate
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )

    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer)
    
    gru_output = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)))(embedding_layer)
    # gru_output = Dropout(0.5)(gru_output)
    attention_output = MultiHeadAttention(num_heads=1, key_dim=100)(gru_output, gru_output)
    attention_output = Dropout(0.5)(attention_output)
    attention_output = attention_output[:, -1, :] 
    output_layer = Dense(1, activation='sigmoid')(attention_output)  # Final output layer for binary classification

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',     # Monitor validation loss
        factor=0.5,             # Factor to reduce the learning rate
        patience=5,             # Number of epochs with no improvement to wait
        min_lr=1e-6             # Minimum learning rate limit
    )
    # Set optimizer
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, checkpoint_callback]
    )

    return model, history
if not Flag or Flag == 3:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (biGRU with attention): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (biGRU with attention): ", accuracy[1], file=file)


def add_and_norm(x, sublayer_output):
    if x.shape[-1] != sublayer_output.shape[-1]:
        # Project `sublayer_output` to match `x`'s shape
        sublayer_output = Dense(x.shape[-1])(sublayer_output)
        
    # Add and normalize
    add_output = Add()([x, sublayer_output])
    norm_output = LayerNormalization()(add_output)
    return norm_output

#EVERYTHING WITH ADD and NORM around 75% with schedule learning and no early stopping
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )

    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer)
    
    gru_output = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)))(embedding_layer)
    gru_output = Dropout(0.5)(gru_output)
    attention_output = MultiHeadAttention(num_heads=1, key_dim=100)(gru_output, gru_output)
    attention_output = Dropout(0.5)(attention_output)
    attention_output = attention_output[:, -1, :] 
    gru_output = gru_output[:, -1, :]
    attention_with_residual = add_and_norm(gru_output, attention_output)
    # attention_output = attention_output[:, -1, :]
    # max_output = GlobalMaxPooling1D()(attention_output)
    # mean_output = GlobalAveragePooling1D()(attention_output)
    # gru_output = gru_output[:, -1, :]  # Get the last output of the GRU layer
    # concat_output = tf.keras.layers.Concatenate(axis=1)([max_output, mean_output, gru_output])  # Concatenate the outputs of the two layers 
    dense_output = Dense(50, activation='relu')(attention_with_residual)
    dense_output = Dropout(0.5)(dense_output)
    dense_with_residual = add_and_norm(attention_with_residual, dense_output)

    output_layer = Dense(1, activation='sigmoid')(dense_with_residual)  # Final output layer for binary classification

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',     # Monitor validation loss
        factor=0.5,             # Factor to reduce the learning rate
        patience=5,             # Number of epochs with no improvement to wait
        min_lr=1e-6             # Minimum learning rate limit
    )
    # Set optimizer
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, checkpoint_callback, reduce_lr]
    )

    return model, history


if not Flag or Flag == 3:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (final): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (final): ", accuracy[1], file=file)


def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )

    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer)
    
    gru_output = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)))(embedding_layer)
    gru_output = Dropout(0.5)(gru_output)
    attention_output = MultiHeadAttention(num_heads=1, key_dim=100)(gru_output, gru_output)
    attention_output = Dropout(0.5)(attention_output)
    attention_output = attention_output[:, -1, :] 
    gru_output = gru_output[:, -1, :]
    attention_with_residual = add_and_norm(gru_output, attention_output)
    # attention_output = attention_output[:, -1, :]
    # max_output = GlobalMaxPooling1D()(attention_output)
    # mean_output = GlobalAveragePooling1D()(attention_output)
    # gru_output = gru_output[:, -1, :]  # Get the last output of the GRU layer
    # concat_output = tf.keras.layers.Concatenate(axis=1)([max_output, mean_output, gru_output])  # Concatenate the outputs of the two layers 
    dense_output = Dense(50, activation='relu')(attention_with_residual)
    dense_output = Dropout(0.5)(dense_output)
    dense_with_residual = add_and_norm(attention_with_residual, dense_output)

    #Parallel CNN
    cnn_output1 = Convolution1D(100, kernel_size=5, strides = 2, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(embedding_layer)
    maxpool_output1 = MaxPooling1D(pool_size=5, strides=2)(cnn_output1)
    maxpool_output1_d = Dropout(0.5)(maxpool_output1)
    cnn_output2 = Convolution1D(50, kernel_size=3, strides = 1, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(maxpool_output1_d)
    maxpool_output2 = MaxPooling1D(pool_size=3, strides=2)(cnn_output2)
    maxpool_output2_d = Dropout(0.5)(maxpool_output2)
    cnn_skip1 = Convolution1D(50, kernel_size=1, activation='relu')(maxpool_output1)
    skip_connection = tf.keras.layers.Concatenate(axis=1)([cnn_skip1, maxpool_output2_d])
    cnn_output3 = Convolution1D(25, kernel_size=3, strides = 1, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(skip_connection)
    maxpool_output3 = MaxPooling1D(pool_size=2, strides=1)(cnn_output3)
    maxpool_output3_d = Dropout(0.5)(maxpool_output3)
    flatten_output = Flatten()(maxpool_output3_d)
    linear_output = Dense(50, activation='relu')(flatten_output)
    linear_output = Dropout(0.5)(linear_output)

    concat_output = tf.keras.layers.Concatenate(axis=1)([linear_output, dense_with_residual])
    output_layer = Dense(1, activation='sigmoid')(concat_output)  # Final output layer for binary classification

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',     # Monitor validation loss
        factor=0.5,             # Factor to reduce the learning rate
        patience=5,             # Number of epochs with no improvement to wait
        min_lr=1e-6             # Minimum learning rate limit
    )
    # Set optimizer
    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, checkpoint_callback, reduce_lr]
    )

    return model, history


if not Flag or Flag == 3:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (parallel): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (parallel): ", accuracy[1], file=file)



# --------------------------- TRANSFORMER -------------------------------
@tf.keras.utils.register_keras_serializable()
class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = Dense(ff_dim, activation='relu')
        self.ffn_output = Dense(embed_dim)
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            "embed_dim": self.attention.key_dim,
            "num_heads": self.attention.num_heads,
            "ff_dim": self.ffn.units,
            "dropout_rate": self.dropout1.rate,
        })
        return config
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )

    num_layers = 2 

    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer)

    # Transformer Encoder Stack
    x = embedding_layer
    for _ in range(num_layers):
        x = TransformerEncoderBlock(embed_dim=embedding_dim, num_heads=2, ff_dim=16, dropout_rate=0.5)(x)

    x = x[:, -1, :]
    # Final layers for classification
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    # Model definition
    model = Model(inputs=input_layer, outputs=x)

    if optimizer == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd': optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',     # Monitor validation loss
        factor=0.5,             # Factor to reduce the learning rate
        patience=5,             # Number of epochs with no improvement to wait
        min_lr=1e-6             # Minimum learning rate limit
    )
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, checkpoint_callback, reduce_lr]
    )

    return model, history

if not Flag or Flag == 3:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (transformer): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (transformer): ", accuracy[1], file=file)