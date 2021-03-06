#/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
from itertools import groupby
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.contrib.layers.python.layers import fully_connected, convolution2d

import tensorflow as tf
import sys

image_dir='/Users/hitmoon/DOGS-images/Images'
image_filenames = glob.glob('{dir}/{glob}'.format(dir=image_dir, glob='n02*/*.jpg'))
#print(image_filenames[0:2])

# 依据品种对图像进行分组
def group_image(filenames):

    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)

    # 将文件名分解为品种和相应的文件名，品种对应于文件夹名称
    image_filename_with_breed = map(lambda filename:
    (filename.split('/')[5], filename), filenames)
    #print(image_filename_with_breed)

    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        # 将每个品种的20% 划入到测试集中
        for i, breed_images in enumerate(breed_images):
            if i % 5 == 0:
                testing_dataset[dog_breed].append(breed_images[1])
            else:
                training_dataset[dog_breed].append(breed_images[1])


        # 检查每个品种的测试集图像是否至少有全部图像的18%
        breed_training_count = len(training_dataset[dog_breed])
        breed_testing_count = len(testing_dataset[dog_breed])
        assert round(breed_testing_count / (breed_training_count + breed_testing_count), 2) > 0.18, "Not enough testing images."
    return training_dataset, testing_dataset

#print(testing_dataset['n02085620-Chihuahua'])


def write_records_file(dataset, record_location):
    '''
    用dataset中的图像填充一个TFRecord文件，并将类别包含进来

    参数:
    dataset : dict(list)
        这个字典的键对应于其值中文件名列表对应的标签
    
    record_location: str
        存储TFRecord的输出路径
    '''

    current_index = 0
    writer = None
    # 枚举dataset, 因为当前索引用于对文件的划分，每隔100幅图像，训练样本的信息就被写入一个新的TFRecord文件中，以加快操作进程
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = '{record_location}.{index}.tfrecords'.format(record_location = record_location, index= current_index)
                print("record_filename = ", record_filename)
                writer = tf.python_io.TFRecordWriter(record_filename)

            # 利用PIL 打开文件
            image = Image.open(image_filename)

            # 转换为灰度图会减少计算量和内存占用，但这并不是必须的
            image = image.convert('L')

            image_bytes = image.resize((250,151)).tobytes()
            # 将标签按照字符串存储比较高效
            image_label = breed.encode('utf-8')

            example = tf.train.Example(features = tf.train.Features(feature = {
                    'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                    'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))}))
            writer.write(example.SerializeToString())

            current_index += 1
    if writer:
        writer.close()



def encode_to_tf_records():

    training_dataset, testing_dataset = group_image(image_filenames)
    write_records_file(testing_dataset, './output/testing-images/testing-image')
    write_records_file(training_dataset, './output/training-images/training-image')


def decode_tf_records():
    # 读取保存的TFRecord记录文件
    print("reading TFRecords files ...")
    filenames = glob.glob('./output/training-images/*.tfrecords')
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
    serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })

    record_image = tf.decode_raw(features['image'], tf.uint8)
    # 修改图像有助于训练和输出的可视化
    image = tf.reshape(record_image, [250, 151, 1])
    label = tf.cast(features['label'], tf.string)
    print("get image and label")
    print("label = ", label)
    return image,label

def get_batch(batch_size):

    image, label = decode_tf_records()

    # 获取一个batch
    print("get a batch")
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size = batch_size,
    capacity = 50000, min_after_dequeue = 10000, num_threads = 3)

    # 将图像转换为灰度值位于[0,1)的浮点数，与convolution2d 期望的输入匹配
    float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

    return float_image_batch, label_batch

def define_graph(input_image, batch_size):
    print("string define the graph...")

    # 卷积层1
    conv2d_layer_1 = convolution2d(
    input_image,
    num_outputs = 32, # filter 个数
    kernel_size = (5,5),      # filter 的宽和高
    activation_fn = tf.nn.relu,
    weights_initializer = tf.random_normal_initializer,
    stride = [2, 2],
    trainable = True)

    pool_layer_1 = tf.nn.max_pool(conv2d_layer_1, ksize = [1, 2, 2, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'SAME')

    # 卷积层2
    conv2d_layer_2 = convolution2d(
    pool_layer_1,
    num_outputs = 64,
    kernel_size = (5, 5),
    activation_fn = tf.nn.relu,
    weights_initializer = tf.random_normal_initializer,
    stride = (1, 1),
    trainable = True)

    pool_layer_2 = tf.nn.max_pool(conv2d_layer_2, ksize = [1, 2, 2, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'SAME')

    flattened_layer_2 = tf.reshape(pool_layer_2, [ batch_size, -1 ])

    # weight_init 参数也可以接收一个可调用参数，这里使用的了一个lambda 表达式返回了一个截断的正态分布，并指定了标准差
    hidden_layer_3 = fully_connected(
    flattened_layer_2,
    512,
    weights_initializer = lambda i, dtype, partition_info: tf.truncated_normal([38912, 512], stddev = 0.1),
    activation_fn = tf.nn.relu
    )

    # 对一些神经元进行dropout，削减他们在模型中的重要性
    hidden_layer_3 = tf.nn.dropout(hidden_layer_3, 0.1)

    # 输出是前面的层与训练中可用的120个不同狗品种的全连接
    final_fully_connected = fully_connected(
    hidden_layer_3,
    120, # 120 种狗
    weights_initializer = lambda i, dtype, partition_info: tf.truncated_normal([512, 120], stddev = 0.1)
    )
    print("graph is ready")
    return final_fully_connected


def train_rec_dog():

    # encode_to_tf_records()

    image_batch, label_batch = get_batch(3)

    # 找出所有狗的品种
    labels = list(map(lambda c: c.split('/')[-1], glob.glob('{dir}/*'.format(dir=image_dir))))
    # 匹配每个来自label_batch的标签并返回他们在类别列表中的索引
    train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0, 0:1][0], label_batch, dtype=tf.int64)

    out = define_graph(image_batch, 3)

    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=train_labels, logits = out))

    batch = tf.Variable(0)
    '''
    learning_rate = tf.train.exponential_decay(
    0.001,
    batch * 3,
    120,
    0.95,
    staircase=True)
    '''

    train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss, global_step = batch)
    #print('train_labels shape =', train_labels.get_shape())
    #print('fully_connected shape =', final_fully_connected.get_shape())
    train_prediction = tf.nn.softmax(out)
    predict = tf.argmax(train_prediction, 1)
    corr = tf.equal(train_labels, predict)
    acc = tf.reduce_mean(tf.cast(corr, tf.float32))

    steps = 1000

    print("start training ...")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)

        for step in range(steps):
            #print('training ...')
            sess.run(train_op)
            if step % 10 == 0:
                print("step:", step, "loss:", sess.run(loss), "label", sess.run(train_labels), "predict", sess.run(predict), "accuracy:", sess.run(acc))

        coord.request_stop()
        coord.join(threads)
        print("train done!")
        sess.close()


if __name__ == '__main__':
    print("start .....")
    train_rec_dog()
