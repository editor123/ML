#/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
image_filenames = glob.glob('/home/hitmoon/DOGS-images/Images/n02*/*.jpg')
#print(image_filenames[0:2])

from itertools import groupby
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import sys

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# 将文件名分解为品种和相应的文件名，品种对应于文件夹名称
image_filename_with_breed = map(lambda filename:
(filename.split('/')[5], filename), image_filenames)
#print(image_filename_with_breed)


# 依据品种对图像进行分组
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



write_records_file(testing_dataset, './output/testing-images/testing-image')
write_records_file(training_dataset, './output/training-images/training-image')
sys.exit(1)


filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('.output/training-images/*.tfrecords'))
reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)

features = tf.parse_single_example(
seriaalized,
features={
    'label': tf.FixedLenFeature([], tf.string),
    'image': tf.FixedLenFeature([], tf.string),
})

record_image = tf.decode_raw(features['image'], tf.uint8)
# 修改图像有助于训练和输出的可视化
image = tf.reshape(record_image, [250, 151, 1])
label = tf.cast(features['label'], tf.string)
min_after_dequeue = 10
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size

# 获取一个batch
image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size = batch_size,
capacity = capacity, min_after_dequeue = min_after_dequeue)

# 将图像转换为灰度值位于[0,1)的浮点数，与convolution2d 期望的输入匹配
float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

# 卷积层1
conv2d_layer_1 = tf.contrib.layers.convolution2d(
float_image_batch,
num_output_channels = 32, # filter 个数
kernel_size = (5,5),      # filter 的宽和高
activation_fn = tf.nn.relu,
weight_init = tf.random_normal,
stride = (2, 2),
trainable = True)

pool_layer_1 = tf.nn.max_pool(conv2d_layer_1, ksize = [1, 2, 2, 1],
                strides = [1, 2, 2, 1],
                padding = 'SAME')

# 卷积层2

conv2d_layer_2 = tf.contrib.layers.convolution2d(
pool_layer_1,
num_output_channels = 64,
kernel_size = (5, 5),
activation_fn = tf.nn.relu,
stride = (1, 1),
trainable = True)

pool_layer_2 = tf.nn.max_pool(conv2d_layer_2, ksize = [1, 2, 2, 1],
                strides = [1, 2, 2, 1],
                padding = 'SAME')

flattened_layer_2 = tf.reshape(pool_layer_2, [ batch_size, -1 ])

# weight_init 参数也可以接收一个可调用参数，这里使用的了一个lambda 表达式返回了一个截断的正态分布，并指定了标准差
hidden_layer_3 = tf.contrib.layers.fully_connected(
flattened_layer_2,
512,
weight_init = lambda i, dtype: tf.truncated_normal([38912, 512], stddev = 0.1),
activation_fn = tf.nn.relu
)

# 对一些神经元进行dropout，削减他们在模型中的重要性
hidden_layer_3 = tf.nn.dropout(hidden_layer_3, 0.1)

# 输出是前面的层与训练中可用的120个不同狗品种的全连接
final_fully_connected = tf.contrib.layers.fully_connected(
hidden_layer_3,
120, # 120 种狗
weight_init = lambda i, dtype: tf.truncated_normal([512, 120], stddev = 0.1)
)


                

