import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding

from keras.optimizers import RMSprop

from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import random
import sys

with open('input.txt', 'r', encoding = "utf8") as file:
    text = file.read()

words = text.split(' ')
vocabulary = sorted(list(set(words)))

#words_print = ' '.join("'{0}'".format(x) for x in words)
#with open('words.txt', 'w', encoding = 'utf-8') as file:
    #file.write(words_print)

word_to_idx = {w: i for i, w in enumerate(vocabulary)}
idx_to_word = {i: w for i, w in enumerate(vocabulary)}

sequences = [word_to_idx[word] for word in words if word in word_to_idx]

max_length = 10
steps = 1
sentences = []
next_words = []

for i in range(0, len(sequences) - max_length, steps):
    sentences.append(sequences[i: i + max_length])
    next_words.append(sequences[i + max_length])

X = np.zeros((len(sentences), max_length), dtype = np.int32)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t] = word
    y[i, next_words[i]] = 1


model = Sequential()
model.add(Embedding(len(vocabulary), 100, input_length = max_length))
model.add(LSTM(128))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model.fit(X, y, batch_size = 128, epochs = 50)

def generate_text(length, diversity):

    start_index = random.randint(0, len(sequences) - max_length - 1)
    generated = []
    sentence = sequences[start_index: start_index + max_length]
    generated.extend(sentence)
    
    for i in range(length):
        x_pred = np.zeros((1, max_length))
        for t, word in enumerate(sentence):
            x_pred[0, t] = word

        preds = model.predict(x_pred, verbose = 0)[0]
        next_index = sample_index(preds, diversity)
        next_word = next_index
        
        generated.append(next_word)
        sentence = sentence[1:] + [next_word]
    
    return ' '.join([idx_to_word[idx] for idx in generated])

generated_text = generate_text(1000, 0.2)
print(generated_text)
#with open('gen.txt', 'w', encoding='utf-8') as file:
    #file.write(generated_text)