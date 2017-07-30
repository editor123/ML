import tensorflow as tf


w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    save_path = saver.save(sess, "/Users/hitmoon/ML/model_save")
    print ("modele saved in file", save_path)
