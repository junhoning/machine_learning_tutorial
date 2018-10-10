import numpy as np
import tensorflow as tf


height = 28
width = 28
num_channel = 1  # rgb / gray scale
num_classes = 10


# Convolution vs Perceptron
def conv(x, num_channel, out_channel=64, name='_layer_1'):
    w1 = tf.Variable(tf.random_normal([3, 3, num_channel, out_channel], stddev=0.1), name='w' + name)
    b1 = tf.Variable(tf.random_normal([out_channel], stddev=0.1), name='w' + name)
    conv = tf.nn.conv2d(x, filter=w1, strides=[1, 1, 1, 1], padding='SAME', name='conv' + name)

    act_1 = tf.nn.relu(conv, name='act' + name)
    return act_1


def max_pool(layer, name='pool1'):
    pool = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    return pool


# Fully Connected Layer
def fully_conv(layer, name='fc'):
    fc = tf.layers.flatten(layer)
    fc_w = tf.Variable(tf.random_normal([6272, num_classes], stddev=0.1), name='w_' + name)
    fc_b = tf.Variable(tf.random_normal([num_classes], stddev=0.1), name='b_' + name)
    fc_out = tf.nn.bias_add(tf.matmul(fc, fc_w), fc_b)
    return fc_out


def cnn(x):
    conv1 = conv(x, 1, 64, name='_layer_1')
    pool1 = max_pool(conv1, name='pool_1')

    conv2 = conv(pool1, 64, 128, name='_layer_2')
    pool2 = max_pool(conv2, name='pool_2')

    logit = fully_conv(pool2, name='fc')
    return logit


def onehot_encoder(label, num_classes=10):
    onehot = np.zeros([num_classes])
    onehot[label] = 1
    return onehot

