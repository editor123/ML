import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

print ("packs loaded")

print ("Download and Extract MNIST dataset")
mnist = input_data.read_data_sets('data/', one_hot = True)

# what does the data of MNIST look like?

print ("what does the data of MNIST look like?")
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print ("type of trainimg is %s" % (type(trainimg)))
print ("type of trainlabel is %s" % (type(trainlabel)))
print ("type of testimg is %s" % (type(testimg)))
print ("type of testlabel is %s" % (type(testlabel)))


print ("shape of trainimg is %s" % (trainimg.shape,))
print ("shape of trainlabel is %s" % (trainlabel.shape,))
print ("shape of testimg is %s" % (testimg.shape,))
print ("shape of testlabel is %s" % (testlabel.shape,))


# How the training data look like ?

print ("How the training data look like ?")
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
    curr_img = np.reshape(trainimg[i, :], (28, 28)) # 28 * 28 matrix
    curr_label = np.argmax(trainlabel[i, :]) # label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i) + "th Training data Label is " + str(curr_label))
    
    print ("" + str(i) + "th Training data Label is " + str(curr_label))    
    plt.show()

n_input = 784
n_output = 10

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev = 0.1)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.1)),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev = 0.1)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev = 0.1))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev = 0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev = 0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev = 0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev = 0.1))
}


def conv_basic(_input, _w, _b, _keepartio):
    # INPUT
    _input_r = tf.reshape(_input, shape = [-1, 28, 28, 1])

    # CONV layer 1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides = [1, 1, 1, 1], padding = 'SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepartio)

    # CONV layer 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides = [1, 1, 1, 1], padding = 'SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepartio)

    # vectorize
    _densel = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    # fully connected layer 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_densel, _w['wd1']), _b['bd1']))
    _fc1_dr1 = tf.nn.dropout(_fc1, _keepartio)

    # fully connected layer 2
    _out = tf.add(tf.matmul(_fc1_dr1, _w['wd2']), _b['bd2'])

    # return
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'densel': _densel,
           'fc1': _fc1, 'fc_dr1': _fc1_dr1, 'out': _out
    }
    return out

print("CNN Network ready!")

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# functions
_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = _pred, labels = y))
optm = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()

# SAVER
print("GRAPH READY!")

sess = tf.Session()
sess.run(init)

training_epochs = 15
batch_size = 16
display_step = 1

for epoch in range(training_epochs):
    avg_cost = 0
    #total_batch = int(mnist.train.num_examples / batch_size)
    total_batch = 10

    # loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict = {x: batch_xs, y: batch_ys, keepratio: 0.7})
        avg_cost += sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys, keepratio: 1}) / total_batch

    # display logs
    if epoch % display_step == 0:
        print("Epoch: %03d / %03d, cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict = {x: batch_xs, y: batch_ys, keepratio: 1})
        print("Training accuracy: %.3f" % (train_acc))

        test_acc = sess.run(accr, feed_dict = {x: testimg, y: testlabel, keepratio: 1})
        print("Test accuracy: %.3f" % (test_acc))

print("OPTIMIZATION FINISH (CNN) !")
