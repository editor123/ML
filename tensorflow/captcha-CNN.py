
# -*- coding: utf-8 -*-
import numpy as np
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image
import random

import tensorflow as tf
import sys

# 验证码字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

charset = number

CHAR_SET_LEN = len(charset)
IMAGE_HEIGHT = 60
IMAGE_WITDH = 160
MAX_CAPTCHA = 4

# 生成长度为4的随机字符序列
def random_captcha_text(charset = number, captcha_size = 4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(charset)
        captcha_text.append(c)

    return captcha_text

# 生成对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


# 把彩色图转换为灰度图
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        '''
        正规做法如下, 上面做法较快
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        '''
        return gray
    else:
        return img

# 文本转向量
def text2vec(text):
    text_len = len(text)
    if (text_len > MAX_CAPTCHA):
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + int(c)
        vector[idx] = 1

    return vector


# 向量转文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        text.append(str(char_idx))

    return "".join(text)

# 生成一个训练的batch
def get_next_batch(batch_size = 128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WITDH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成的图像大小不是(60 * 160 * 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i,:] = image.flatten() / 255
        batch_y[i,:] = text2vec(text)

    return batch_x, batch_y



X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WITDH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keepratio = tf.placeholder(tf.float32)

# CNN 定义
def crack_captcha_cnn(w_alpha = 0.01, b_alpha = 0.1):
    x = tf.reshape(X, shape = [-1, IMAGE_HEIGHT, IMAGE_WITDH, 1])

    # layer 1
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
    pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    dr1 = tf.nn.dropout(pool1, keepratio)


    # layer 2
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dr1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
    pool2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    dr2 = tf.nn.dropout(pool2, keepratio)

    # layer 3
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3= tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dr2, w_c3, strides = [1, 1, 1, 1], padding = 'SAME'), b_c3))
    pool3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    dr3 = tf.nn.dropout(pool3, keepratio)

    # fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(dr3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keepratio)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

# 训练
def train_crack_captcha_cnn():

    output = crack_captcha_cnn()

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output, labels = Y))

    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict = {X: batch_x, Y: batch_y, keepratio: 0.75})

            # 每迭代10次输出一次loss
            if step % 10 == 0:
                print("step: %d, loss: %03f" % (step, loss_))

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc_ = sess.run(acc, feed_dict = {X: batch_x_test, Y: batch_y_test, keepratio: 1.})
                print("accuracy: %09f" % acc_)

                # 如果准确率大于 50 %, 保存模型， 完成训练
                if acc_ > 0.96:
                    saver.save(sess, "crack_captcha.model", global_step = step)
                    break

            step += 1

# 用模型识别验证码
def crack_captcha(captcha_image):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './crack_captcha.model-2500')

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict = {X: [captcha_image], keepratio:1})
        text = text_list[0].tolist()
        return text

if __name__ == '__main__':

    '''
    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha = 'center', va = 'center', transform = ax.transAxes)
    plt.imshow(image)
    plt.show()

    print("验证码图像Channel:", image.shape)

    vector = text2vec(text)
    print (vector)
    t = vec2text(vector)
    print (t)


    print("string training ...")
    train_crack_captcha_cnn();
    print("traing finished !")
    '''
    if len(sys.argv) == 2:
        command = 'c'
    else:
        command = 't'

    if command == 'c':
        print("continue training...")
        train_crack_captcha_cnn();

    else:
        print("Let's test the model ...")
        # predict
        text, image = gen_captcha_text_and_image()
        image = convert2gray(image)
        image = image.flatten() / 255
        predict_text = crack_captcha(image)

        print("正确: {}, 预测: {}".format(text, predict_text))

