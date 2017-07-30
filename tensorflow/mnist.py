import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

print ("packs loaded")

print ("Download and Extract MNIST dataset")
mnist = input_data.read_data_sets('data/', one_hot = True)

print ("type of 'mnist' is %s " % (type(mnist)))
print ("number of train data is %d" % (mnist.train.num_examples))
print ("number of test data is %d" % (mnist.test.num_examples))

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

# Batch Learning ?
print ("Batch learning ?")

batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print ("type of 'batch_xs' is %s" % (type(batch_xs)))
print ("type of 'batch_ys' is %s" % (type(batch_ys)))
print ("shape of 'batch_xs' is %s" % (batch_xs.shape,))
print ("shape of 'batch_ys' is %s" % (batch_ys.shape,))


# Network topologies

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10


# input and output
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Network parameters
stddev = 0.1
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

print ("Network ready!")

def multilayer_perceptron(_x, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_x, _weights['w1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    result = tf.matmul(layer_2, _weights['out']) + _biases['out']
    return result


# prediction
pred = multilayer_perceptron(x, weights, biases)

# loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

# initializer

init = tf.global_variables_initializer()
print ("functions ready!")

training_epochs = 20
batch_size = 100
display_step = 4

# launch the graph
sess = tf.Session()
sess.run(init)

# optimize

print ("start optimization ...")
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    # iteration
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict = feeds)
        avg_cost += sess.run(cost, feed_dict = feeds)

    avg_cost = avg_cost / total_batch

    # display

    if (epoch + 1) % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict = feeds)
        print ("training accuracy: %.3f" % train_acc)
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict = feeds)
        print ("test accuracy: %.3f" % test_acc)


print ("optimization finished!")
        
