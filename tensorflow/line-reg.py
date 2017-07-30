
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 1000 random points around y= 0.1x + 0.3

num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1,y1])

# generate samples

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


# 生成一维的 w 矩阵， 取值是[-1, 1]之间的随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name = 'W')
# 生成一维的 b 矩阵， 初始值是0
b = tf.Variable(tf.zeros(1), name = 'b')
# 经过计算得出预估值 y
y = W * x_data + b

# 以预估值y 和 y_data 之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data), name = 'loss')

# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss, name = 'train')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 输出初始的W 和 b 值 以及 loss
print ('W = ', sess.run(W), 'b = ', sess.run(b), 'loss = ', sess.run(loss))


# 执行20次训练
for step in range(20):
    sess.run(train)
    # 输出 W, b, loss
    print ('W = ', sess.run(W), 'b = ', sess.run(b), 'loss = ', sess.run(loss))

# writer = tf.train.SummaryWriter("./", sess.graph)
