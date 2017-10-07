import numpy as np

from string import ascii_lowercase

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
import keras

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # moving the window over the input series
    for i in range(len(series) - window_size):
        batch = series[i:i + window_size + 1]
        X.append(batch[:-1])
        y.append([batch[-1]])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model

def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    allowed = punctuation + list(ascii_lowercase) + [' ']

    # filtering out all the characters that are not allowed
    text = ''.join([c for c in text if c in allowed])

    return text

def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # moving the window over the text to form input-output pairs
    for i in range(0, len(text) - window_size, step_size):
        batch = text[i:i + window_size + 1]
        inputs.append(batch[:-1])
        outputs.append(batch[-1])

    return inputs, outputs

def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))

    return model
