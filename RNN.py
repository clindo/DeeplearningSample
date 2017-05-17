'''
This File implements a RNN using LTSM
'''
import gzip
import cPickle as pkl
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.utils import np_utils


f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

w2idx = embeddings['word2Idx']
label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']

# Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}
print  idx2Label

f = gzip.open('pkl/data.pkl.gz', 'rb')
train_tokens, train_y = pkl.load(f)
test_tokens, test_y = pkl.load(f)
f.close()

# Create index to word/label dicts
idx2w = {w2idx[k]: k for k in w2idx}
idx2la = {label2Idx[k]: k for k in label2Idx}


# truncate and pad input sequences
max_length = 240
X_train = sequence.pad_sequences(train_tokens, maxlen=max_length)
X_test = sequence.pad_sequences(test_tokens, maxlen=max_length)

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, 12)

# create the model
model = Sequential()
model.add(Embedding(wordEmbeddings.shape[0], wordEmbeddings.shape[1], input_length=max_length))
model.add(LSTM(100))
model.add(Dense(12, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, train_y_cat, epochs=3, batch_size=720)
# Final evaluation of the model
scores = model.evaluate(X_test, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
