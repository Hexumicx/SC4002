# Part 0. Data Prepraration
from datasets import load_dataset
dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset ['train']
validation_dataset = dataset ['validation']
test_dataset = dataset ['test']

#Train what?
# 0 for All
# 1 for Dense Layer
# 2 for CNN
# 6 for Attention + multihead + Label Smmoothing + dropout
# 7 for add on to BiGRU
Flag = 25


# Part 1. Preparing Word Embeddings
# (a) What is the size of the vocabulary formed from your training data?
with open("/app/result/result.txt", "w") as file:
    print("Part 1. Preparing Word Embeddings:", file=file)

import nltk
import numpy as np
nltk.download('punkt')
nltk.download('punkt_tab')

# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# def remove_stopwords(example):
#     example['text'] = ' '.join(
#         [word for word in example['text'].split() if word.lower() not in stop_words]
#     )
#     return example

# train_dataset = train_dataset.map(remove_stopwords)
# validation_dataset = validation_dataset.map(remove_stopwords)
# test_dataset = test_dataset.map(remove_stopwords)

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

if not Flag or Flag == 2:
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

with open("/app/result/result.txt", "a") as file:
    print("--------BiLSTM--------", file=file)

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
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        Bidirectional(LSTM(16, return_sequences=True)),
        Attention(units=16),  # Attention layer to compute context vector
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 6 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Attention): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Attention): ", accuracy[1], file=file)

# Add Dropout
from tensorflow.keras.layers import Dropout

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

    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is trainable
        
        Bidirectional(LSTM(16, return_sequences=False)),  # LSTM layer
        Dropout(0.5),  # Dropout after LSTM
        
        # Dense(16, activation='relu'),  # Intermediate Dense layer with ReLU activation
        # Dropout(0.5),  # Dropout after Dense layer
        
        Dense(1, activation='sigmoid')  # Final Dense layer for binary classification
    ])

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

# Train the model
if not Flag or Flag ==6 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Dropout): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Dropout): ", accuracy[1], file=file)

# Add MultiHead Attention

from tensorflow.keras.layers import MultiHeadAttention, Input
from tensorflow.keras.models import Model

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

    # Multi-Head Attention requires both query and key-value pairs
    attention_output = MultiHeadAttention(num_heads=2, key_dim=embedding_dim)(embedding_layer, embedding_layer)
    
    # Combine attention output with LSTM
    lstm_output = Bidirectional(LSTM(16, return_sequences=False))(attention_output)
    dropout_output = Dropout(0.5)(lstm_output)  # Add dropout for regularization
    output_layer = Dense(1, activation='sigmoid')(dropout_output)  # Final output layer for binary classification

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # model = Sequential([
    #     Embedding(input_dim=vocab_size,
    #               output_dim=embedding_dim,
    #               weights=[embedding_matrix],
    #               trainable=True),  # Embedding layer is trainable
        
    #     MultiHeadAttention(num_heads=2, key_dim=embedding_dim),  # Multi-Head Attention layer
        
    #     Bidirectional(LSTM(16, return_sequences=False)),  # LSTM layer
    #     Dense(1, activation='sigmoid')  # Final Dense layer for binary classification
    # ])

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

# Train the model
if not Flag or Flag ==6 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added MultiHead Attention): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added MultiHead Attention): ", accuracy[1], file=file)

# Add Adversarial Training
#To do

# Add label Smoothing
def train_model(optimizer, epochs, batch_size, lr, label_smoothing=0.1):
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

    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is trainable
        Bidirectional(LSTM(16, return_sequences=False)),
        Dense(1, activation='sigmoid')
    ])

    # Set optimizer
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    
    # Set loss with label smoothing
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, checkpoint_callback]
    )

    return model, history

# Train the model with label smoothing
if not Flag or Flag ==6 or Flag == 20:
    model, history = train_model("adagrad", epochs=10, batch_size=64, lr=0.01, label_smoothing=0.1)
    
    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Label Smoothing 0.1): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Label Smoothing 0.1): ", accuracy[1], file=file)

with open("/app/result/result.txt", "a") as file:
    print("--------BiGRU--------", file=file)

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
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        Bidirectional(GRU(16, return_sequences=True)),
        Attention(units=16),  # Attention layer to compute context vector
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 7 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Attention): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Attention): ", accuracy[1], file=file)

# Add Dropout
from tensorflow.keras.layers import Dropout

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

    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is trainable
        
        Bidirectional(GRU(16, return_sequences=False)),  # LSTM layer
        Dropout(0.5),  # Dropout after LSTM
        
        # Dense(16, activation='relu'),  # Intermediate Dense layer with ReLU activation
        # Dropout(0.5),  # Dropout after Dense layer
        
        Dense(1, activation='sigmoid')  # Final Dense layer for binary classification
    ])

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

# Train the model
if not Flag or Flag == 7 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Dropout): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Dropout): ", accuracy[1], file=file)

# Add MultiHead Attention

from tensorflow.keras.layers import MultiHeadAttention, Input
from tensorflow.keras.models import Model

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

    # Multi-Head Attention requires both query and key-value pairs
    attention_output = MultiHeadAttention(num_heads=2, key_dim=embedding_dim)(embedding_layer, embedding_layer)
    
    # Combine attention output with GRU
    lstm_output = Bidirectional(GRU(16, return_sequences=False))(attention_output)
    dropout_output = Dropout(0.5)(lstm_output)  # Add dropout for regularization
    output_layer = Dense(1, activation='sigmoid')(dropout_output)  # Final output layer for binary classification

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # model = Sequential([
    #     Embedding(input_dim=vocab_size,
    #               output_dim=embedding_dim,
    #               weights=[embedding_matrix],
    #               trainable=True),  # Embedding layer is trainable
        
    #     MultiHeadAttention(num_heads=2, key_dim=embedding_dim),  # Multi-Head Attention layer
        
    #     Bidirectional(LSTM(16, return_sequences=False)),  # LSTM layer
    #     Dense(1, activation='sigmoid')  # Final Dense layer for binary classification
    # ])

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

# Train the model
if not Flag or Flag == 7 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added MultiHead Attention): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added MultiHead Attention): ", accuracy[1], file=file)

# Add Adversarial Training
#To do

# Add label Smoothing
def train_model(optimizer, epochs, batch_size, lr, label_smoothing=0.1):
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

    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is trainable
        Bidirectional(GRU(16, return_sequences=False)),
        Dense(1, activation='sigmoid')
    ])

    # Set optimizer
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    
    # Set loss with label smoothing
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[custom_callback, checkpoint_callback]
    )

    return model, history

# Train the model with label smoothing
if not Flag or Flag == 7 or Flag == 20:
    model, history = train_model("adagrad", epochs=10, batch_size=64, lr=0.01, label_smoothing=0.1)
    
    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Label Smoothing 0.1): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Label Smoothing 0.1): ", accuracy[1], file=file)




# ------------------------------------------ MOREEEEEEEEEEE ------------------------------
from tensorflow.keras.layers import Reshape
# bigru - bigru - attention
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
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
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
        Bidirectional(GRU(32, return_sequences=True)),
        Bidirectional(GRU(16, return_sequences=True)),
        Attention(units=16),  # Attention layer to compute context vector
        Dropout(0.2), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Attention, dropout 0.2, BiGRU 32, 16): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Attention): ", accuracy[1], file=file)

# ----------------------------------------------- Plus masking -------------------------------------------------------------
from tensorflow.keras.layers import Masking

# bigru - bigru - attention MASK
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_states, mask=None):
        # score = tf.nn.tanh(self.W(hidden_states))
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector = attention_weights * hidden_states
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        # return context_vector
        score = tf.nn.tanh(self.W(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        if mask is not None:
            # Expand the mask to match attention_weights shape
            mask = tf.cast(mask, dtype=attention_weights.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            attention_weights *= mask  # Apply mask

        attention_weights /= tf.reduce_sum(attention_weights, axis=1, keepdims=True)  # Renormalize

        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
        
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Masking(mask_value=0), 
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  
        Bidirectional(GRU(64, return_sequences=True)),
        Attention(units=64),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 12 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Masking): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Masking): ", accuracy[1], file=file)

# bigru - bigru - attention MASK LSTM
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_states, mask=None):
        # score = tf.nn.tanh(self.W(hidden_states))
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector = attention_weights * hidden_states
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        # return context_vector
        score = tf.nn.tanh(self.W(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        if mask is not None:
            # Expand the mask to match attention_weights shape
            mask = tf.cast(mask, dtype=attention_weights.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            attention_weights *= mask  # Apply mask

        attention_weights /= tf.reduce_sum(attention_weights, axis=1, keepdims=True)  # Renormalize

        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
        
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Masking(mask_value=0), 
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  
        Bidirectional(LSTM(16, return_sequences=True)),
        Attention(units=16),  # Attention layer to compute context vector
        Dropout(0.25), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 13 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Masking LSTM): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Masking LSTM): ", accuracy[1], file=file)        

# bigru - attention - bigru
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
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
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
        Bidirectional(GRU(32, return_sequences=True)),
        Attention(units=32),
        Reshape((-1, 32)),
        Bidirectional(GRU(16, return_sequences=False)),
        Dropout(0.2), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 11 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (bigru - attention - bigru): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (bigru - attention - bigru): ", accuracy[1], file=file)








############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################
############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################
############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################
############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################
############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################
############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################
############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################
############ BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ########################################## BREAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ##############################





@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        # This method initializes the layer's state based on the input shape
        super().build(input_shape)    

    def call(self, hidden_states, mask=None):
        # score = tf.nn.tanh(self.W(hidden_states))
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector = attention_weights * hidden_states
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        # return context_vector
        score = tf.nn.tanh(self.W(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        if mask is not None:
            # Expand the mask to match attention_weights shape
            mask = tf.cast(mask, dtype=attention_weights.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            attention_weights *= mask  # Apply mask

        attention_weights /= tf.reduce_sum(attention_weights, axis=1, keepdims=True)  # Renormalize

        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
        
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        Masking(mask_value=0), 
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  mask_zero=True,
                  trainable=True),  
        GRU(embedding_dim, return_sequences=True),
        Attention(units=embedding_dim),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 15 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (dimension == embedding_dim): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (dimension == embedding_dim): ", accuracy[1], file=file)
#0.78893



@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        # This method initializes the layer's state based on the input shape
        super().build(input_shape)    

    def call(self, hidden_states, mask=None):
        # score = tf.nn.tanh(self.W(hidden_states))
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector = attention_weights * hidden_states
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        # return context_vector
        score = tf.nn.tanh(self.W(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        if mask is not None:
            # Expand the mask to match attention_weights shape
            mask = tf.cast(mask, dtype=attention_weights.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            attention_weights *= mask  # Apply mask

        attention_weights /= tf.reduce_sum(attention_weights, axis=1, keepdims=True)  # Renormalize

        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
        
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        # Masking(mask_value=0), 
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                #   mask_zero=True,
                  trainable=True),  
        GRU(embedding_dim, return_sequences=True),
        Attention(units=embedding_dim),  # Attention layer to compute context vector
        Dropout(0.2), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 16 or Flag == 20:
    model, history = train_model("adagrad", 50, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (dimension == embedding_dim): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (dimension == embedding_dim): ", accuracy[1], file=file)



@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        # This method initializes the layer's state based on the input shape
        super().build(input_shape)    

    def call(self, hidden_states, mask=None):
        # score = tf.nn.tanh(self.W(hidden_states))
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector = attention_weights * hidden_states
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        # return context_vector
        score = tf.nn.tanh(self.W(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        if mask is not None:
            # Expand the mask to match attention_weights shape
            mask = tf.cast(mask, dtype=attention_weights.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            attention_weights *= mask  # Apply mask

        attention_weights /= tf.reduce_sum(attention_weights, axis=1, keepdims=True)  # Renormalize

        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
        
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        # Masking(mask_value=0), 
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  mask_zero=True,
                  trainable=True),  
        Bidirectional(GRU(embedding_dim, return_sequences=True)),
        Attention(units=embedding_dim),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 17 or Flag == 20:
    model, history = train_model("adagrad", 50, 128, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (dimension == embedding_dim): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (dimension == embedding_dim): ", accuracy[1], file=file)

#0.7917448282241821


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
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        Bidirectional(LSTM(embedding_dim, return_sequences=True)),
        Attention(units=embedding_dim*2),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag or Flag == 22 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (biLSTM embedding): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (biLSTM embedding): ", accuracy[1], file=file)

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
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        LSTM(embedding_dim, return_sequences=True),
        Attention(units=embedding_dim),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag or Flag == 22 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (LSTM embedding): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (LSTM embedding): ", accuracy[1], file=file)

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
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        Bidirectional(GRU(embedding_dim, return_sequences=True)),
        Attention(units=embedding_dim*2),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag or Flag == 22 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (biGRU embedding): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (biGRU embedding): ", accuracy[1], file=file)


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
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        GRU(embedding_dim, return_sequences=True),
        Attention(units=embedding_dim*2),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag or Flag == 22 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (GRU embedding): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (GRU embedding): ", accuracy[1], file=file)


from tensorflow.keras import regularizers
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        # This method initializes the layer's state based on the input shape
        super().build(input_shape)   

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
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  trainable=True),  # Embedding layer is frozen
        Bidirectional(LSTM(embedding_dim, return_sequences=True)),
        GlobalAveragePooling1D(), 
        # Dense(embedding_dim, activation='relu'),
        # Attention(units=embedding_dim),  # Attention layer to compute context vector
        Dropout(0.5),
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
        callbacks=[custom_callback, checkpoint_callback, early_stopping]
    )
    return model, history

if not Flag or Flag == 23 or Flag == 20:
    model, history = train_model("adagrad", 100, 64, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (RNN embedding): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (RNN embedding): ", accuracy[1], file=file)


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
        Bidirectional(LSTM(1024, return_sequences=True)),
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
        callbacks=[checkpoint_callback, custom_callback, early_stopping]
    )
    return model, history
if not Flag or Flag == 23:
    model, history = train_model("adagrad", 100, 64, 0.01)
    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy for training (Max Pooling):", best_accuracy, file=file)

    best_model = tf.keras.models.load_model("model_max.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("2(c) Best Accuracy on test Set (Max Pooling):", accuracy[1], file=file)



# bigru - bigru - attention
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        # This method initializes the layer's state based on the input shape
        super().build(input_shape)   

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
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=30,
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
        Bidirectional(GRU(embedding_dim, return_sequences=True, kernel_regularizer=regularizers.l1(0.0001))),
        Dropout(0.7), 
        Attention(units=embedding_dim),
        Reshape((-1, embedding_dim)),
        Dropout(0.7),
        Bidirectional(GRU(embedding_dim, return_sequences=True, kernel_regularizer=regularizers.l1(0.0001))),
        Dropout(0.7),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # model = Sequential([
    #     # Masking(mask_value=0), 
    #     Embedding(input_dim=vocab_size,
    #               output_dim=embedding_dim,
    #               weights=[embedding_matrix],
    #               mask_zero=True,
    #               trainable=True),  
    #     Bidirectional(GRU(embedding_dim, return_sequences=True)),
    #     Attention(units=embedding_dim),  # Attention layer to compute context vector
    #     Dropout(0.5), 
    #     Dense(1, activation='sigmoid')
    # ])
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


if not Flag or Flag == 25 or Flag == 20:
    model, history = train_model("adagrad", 500, 128, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (Added Attention, dropout 0.2, BiGRU 32, 16): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (Added Attention): ", accuracy[1], file=file)


@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        # This method initializes the layer's state based on the input shape
        super().build(input_shape)    

    def call(self, hidden_states, mask=None):
        # score = tf.nn.tanh(self.W(hidden_states))
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector = attention_weights * hidden_states
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        # return context_vector
        score = tf.nn.tanh(self.W(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        if mask is not None:
            # Expand the mask to match attention_weights shape
            mask = tf.cast(mask, dtype=attention_weights.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            attention_weights *= mask  # Apply mask

        attention_weights /= tf.reduce_sum(attention_weights, axis=1, keepdims=True)  # Renormalize

        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
        
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config
    
def train_model(optimizer, epochs, batch_size, lr):
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    custom_callback = CustomCallback()
    custom_callback.set_key(optimizer, batch_size, lr)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_combined.keras", 
        monitor='val_accuracy',            
        save_best_only=True,           
        mode='max',                 
        save_weights_only=False,       
        verbose=1
    )
    model = Sequential([
        # Masking(mask_value=0), 
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  mask_zero=True,
                  trainable=True),  
        Bidirectional(GRU(embedding_dim, return_sequences=True)),
        Attention(units=embedding_dim),  # Attention layer to compute context vector
        Dropout(0.5), 
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
        callbacks=[custom_callback, checkpoint_callback]
    )
    return model, history


if not Flag or Flag == 10 or Flag == 26 or Flag == 20:
    model, history = train_model("adagrad", 50, 128, 0.01)

    best_model = tf.keras.models.load_model("model_combined.keras")
    accuracy = best_model.evaluate(X_test, y_test)
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy for training (dimension == embedding_dim): ", best_accuracy, file=file)
    print("Test accuracy:", accuracy[1])
    with open("/app/result/result.txt", "a") as file:
        print("3.5 Best Accuracy on test Set (dimension == embedding_dim): ", accuracy[1], file=file)
