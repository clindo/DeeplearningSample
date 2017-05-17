"""
This file counts the number of sentences and counts the  number of different POS tags
"""

# File which is processed
file = 'data/en_1000'

# At which column position is the token and the tag, starting at 0
tokenPosition = 0
tagPosition = 1

'''
This Function returns the different sentences present in the provided file
'''


def getSentences(filepath, tokenPosition, tagPosition):
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
        sentence.append([splits[tokenPosition].lower(), splits[tagPosition]])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []

    return sentences


'''
This Function count and returns the different Taggs present in the file
'''


def POS_counter(sample_sentences):
    labelSet = set()

    for dataset in [sample_sentences]:
        for sentence in dataset:
            for token, label in sentence:
                labelSet.add(label)
    label_list = list(labelSet)
    return sorted(label_list)


train_set = getSentences(file, tokenPosition, tagPosition)

POS_Taggs = POS_counter(train_set)

print "Number of Sentences is", len(train_set)
print "Number of distinct labels is", len(POS_Taggs)
print "Distinct labels are:", POS_Taggs
