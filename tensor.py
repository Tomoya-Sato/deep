#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from __future__ import print_function

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

#data loading
mnist = fetch_mldata('MNIST original', data_home='./data/')
X, T = mnist.data, mnist.target    # data, label
X = X / 255.
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=2.0)
N_train = X_train.shape[0]
N_test = X_test.shape[0]
T_train = np.eye(10)[T_train.astype("int")]
T_test = np.eye(10)[T_test.astype("int")]


# MLP class definition
def MLP(x):
    layer_1 = tf.layers.dense(x, 1000, tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, 1000, tf.nn.relu)
    out = tf.layers.dense(layer_2, 10, tf.nn.softmax)
    return out


# Graph construction
tf.reset_default_graph()
lr = 0.1    # learning rate
n_epoch = 25
batchsize = 100

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])

y = MLP(x)

xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)
cost = tf.reduce_mean(xentropy)

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Execution
with tf.Session() as sess:
    sess.run(tf.global_Variables_initializer())

    for epoch in range(n_epoch)
        print('epoch %d | ' % epoch, end="")

        # training
        sum_loss = 0
	perm = np.random.permutation(N_train)

	for i in range(0, N_train, batchsize):
            X_batch = X_train[perm[i:i+batchsize]]
            t_batch = T_train[perm[i:i+batchsize]]

            _, loss = sess.run([optimizer, cost], feed_dict={x:X_batch, t:t_batch})
            sum_loss += np.mean(loss) * X_batch.shape[0]

        loss = sum_loss / N_train
        print('Train loss %.5f | ' %(loss), end="")

        print("Test Accuracy: %.3f"%(accuracy.eval(feed_dict={x:X_test, t:T_test})))
