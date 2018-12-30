'''
Credit: youtuber:sentdex 
		video:	 Neural Network Model - Deep Learning with Neural Networks and Tensorflow

program flow:

input > weight > hidden layer 1 (activation function) 
...   > weight > hidden layer 2 (activation function)
...   > output layer

compare output to intended output > cost function 
(cross entropy) optimization function (optimizer)
... > minimize cost (AdamOptimizer....SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''

import tensorflow as tf
from create_sentiment_features import create_feature_sets_and_labels
import numpy as np

# Error logging
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# intialize data
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

n_nodes_hl1 = 500  # hidden layer 1
n_nodes_hl2 = 500  # can be different from other layers
n_nodes_hl3 = 500

n_classes = 2  # probably can be derived
batch_size = 32  # manipulates this number of features in the network

# height x width
x = tf.placeholder('float', [None, len(train_x[0])])  # second arguement creates better error catching
y = tf.placeholder('float')

def neural_network_model(data):

	# bias allows weights to be zero

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),  # generate random weights in tensor(array)
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}  

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),  # generate random weights in tensor(array)
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}  

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),  # generate random weights in tensor(array)
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}  

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),  # generate random weights in tensor(array)
					  'biases':tf.Variable(tf.random_normal([n_classes]))}  


	# (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles of feed forward + backprop
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0

			i = 0
			while i < len(train_x):
				start = i
				end = i+batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size

			print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)


		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)
tf.logging.set_verbosity(old_v)
