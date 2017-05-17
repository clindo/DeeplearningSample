"""
The file preprocesses the data and saves it in Pickle format

"""

import numpy as np
import cPickle as pkl
import gzip

embeddingsPath = 'data/en_embeddings.txt'

folder = 'data/'
files = [folder + 'Training_Data',  folder + 'Test_data']

# At which column position is the token and the tag, starting at 0
tokenPosition = 0
tagPosition = 1

# Size of the context windo
window_size = 3

#This function creates the matrixes for word to Ids and Label to Ids based on window size

def createMatrices(sentences, windowsize, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    xMatrix = []
    yVector = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        for targetWordIdx in xrange(len(sentence)):

            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []
            for wordPosition in xrange(targetWordIdx - windowsize, targetWordIdx + windowsize + 1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    continue

                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1

                wordIndices.append(wordIdx)

            # Get the label and map to int
            labelIdx = label2Idx[sentence[targetWordIdx][1]]
            xMatrix.append(wordIndices)
            yVector.append(labelIdx)

    print "Unknowns: %.2f%%" % (unknownWordCount / (float(wordCount)) * 100)
    return (np.asarray(xMatrix), np.asarray(yVector))


def readFile(filepath, tokenPosition, tagPosition):
    sentences = []
    sentence = []

    for line in open(filepath):
        line = line.strip()

        if len(line) == 0 or line[0] == '#':
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        sentence.append([splits[tokenPosition], splits[tagPosition]])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []

    print filepath, len(sentences), "sentences"
    return sentences


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

outputFilePath = 'pkl/data.pkl.gz'
embeddingsPklPath = 'pkl/embeddings.pkl.gz'

trainSentences = readFile(files[0], tokenPosition, tagPosition)
testSentences = readFile(files[1], tokenPosition, tagPosition)

# Mapping of the labels to integers
labelSet = set()
words = {}

for dataset in [trainSentences, testSentences]:
    for sentence in dataset:
        for token, label in sentence:
            labelSet.add(label)
            words[token.lower()] = True

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = gzip.open(embeddingsPath) if embeddingsPath.endswith('.gz') else open(embeddingsPath)

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]

    if len(word2Idx) == 0:  # Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)

        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)

wordEmbeddings = np.array(wordEmbeddings)

print "Embeddings shape: ", wordEmbeddings.shape
print "Len words: ", len(words)

embeddings = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
              'label2Idx': label2Idx}

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(embeddings, f, -1)
f.close()

# :: Create matrices ::


train_set = createMatrices(trainSentences, window_size, word2Idx, label2Idx)
test_set = createMatrices(testSentences, window_size, word2Idx, label2Idx)

f = gzip.open(outputFilePath, 'wb')
pkl.dump(train_set, f, -1)
pkl.dump(test_set, f, -1)
f.close()

print "Data stored in pkl folder"
