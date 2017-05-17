'''
This file reads the embeddings.pkl.gz file and performs the POS tagging
'''
import numpy as np
import gzip
import cPickle as pkl
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.utils import np_utils
from keras.layers.embeddings import Embedding


f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

numHiddenUnits = 100
label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('pkl/data.pkl.gz', 'rb')
train_tokens, train_y = pkl.load(f)
test_tokens, test_y = pkl.load(f)
f.close()



# Create the  Network
#
#####################################

# Create the train and predict_labels function
n_in = train_tokens.shape[1]
n_out = len(label2Idx)

words = Sequential()
words.add(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], input_length=n_in,  weights=[wordEmbeddings], trainable=False))
words.add(Flatten())

model = Sequential()
model.add(words)

model.add(Dense(units=numHiddenUnits, activation='tanh'))
model.add(Dense(units=n_out, activation='softmax'))


# Use Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)


 # Training of the Network
#
##################################



number_of_epochs = 10
minibatch_size = 128
print "%d epochs" % number_of_epochs


for epoch in xrange(number_of_epochs):
    print "\n------------- Epoch %d ------------" % (epoch+1)
    model.fit(train_tokens, train_y_cat, epochs=1, batch_size=minibatch_size, verbose=True, shuffle=True)


    test_pred = model.predict_classes(test_tokens, verbose=False)
    test_acc = np.sum(test_pred == test_y) / float(len(test_y))
    print "Test-Accuracy: %.2f" % (test_acc*100)
