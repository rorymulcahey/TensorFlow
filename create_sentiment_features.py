'''
Credit: youtuber:sentdex 
		video:	 Neural Network Model - Deep Learning with Neural Networks and Tensorflow

program flow:

Create lexicon from the pos.txt and neg.txt

eg- [chair, table, spoon, television]
"I pulled the chair up to the table"
[1 1 0 0]

'''

# import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter


# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000  # lower number if this creates memory error

def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos, neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)



	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)  # {'the' : 52521, 'and':25424}
	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50:  # don't add very common and rare words
			l2.append(w)

	return l2  # eg: l2 = ['be', 'new', ... 'dull']

def sample_handling(sample, lexicon, classification):
	featureset = []  # eg: featureset = [[0 1 0 1 1 0], [1 0]]; index 0: num instances, index 1: pos or neg

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features, classification])

	return featureset  # eg: featureset = [[0.0, 0.0, ... 1.0, 0.0], [1, 0]]

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1,0])
	features += sample_handling('neg.txt', lexicon, [0,1])
	random.shuffle(features)

	features = np.array(features)  # eg: [[feature0, label0], [feature1, label1], ...]

	testing_size = int(test_size*len(features))

	# first 90%
	train_x = list(features[:,0][:-testing_size])  # eg: [feature0, feature1, ..., feature(x*0.9)]
	train_y = list(features[:,1][:-testing_size])  # eg: [label0, label1, ..., label(x*0.9)]

	# last 10%
	test_x = list(features[:,0][-testing_size:])  # eg: [feature(x*0.9), ..., feature(-1)]
	test_y = list(features[:,1][-testing_size:])  # eg: [feature(x*0.9), ..., label(-1)]

	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)

