import numpy as np
import data_helpers
from w2v import train_word2vec
from sklearn.utils import class_weight
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K

np.random.seed(2)

model_variation = 'CNN-non-static'
print('Model variation is %s' % model_variation)

sequence_length = 45
embedding_dim = 80
filter_sizes = (3, 4)
num_filters = 128
dropout_prob = (0.25, 0.5)
hidden_dims = 128

batch_size = 32
num_epochs = 1
val_split = 0.3

min_word_count = 1  # Minimum word count
context = 12        # Context window size

print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()

if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: {:d}".format(len(vocabulary)))

model = Sequential()
if not model_variation=='CNN-static':
    model.add(Embedding(len(vocabulary), embedding_dim, input_length=119,
                        weights=embedding_weights))

model.add(Conv1D(128, 3, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(128, 4, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(128, 4, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=4))
model.add(GRU(80, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(40, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(20))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x_shuffled, y_shuffled, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split, verbose=1)
model.save("model_review.h5")
