#!/usr/bin/env python3
# -*- coding utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

def generate_conv2d_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        input_x = tf.compat.v1.placeholder(dtype="float32", shape=(1,56,56,64))
        input_filter = tf.compat.v1.placeholder(dtype="float32", shape=(3,3,64,64))
        op = tf.nn.conv2d(input_x, input_filter, strides=[1,1,1,1], padding=[[0,0],[1,1],[1,1],[0,0]],
                          data_format="NHWC", dilations=[1,1,1,1], name='conv2d_res')
        tf.io.write_graph(sess.graph, logdir="./", name="conv2d.pb", as_text=False)

def generate_add_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(tf.int32, name='x')
        y = tf.compat.v1.placeholder(tf.int32, name='y')
        b = tf.Variable(1, name='b')
        xy = tf.multiply(x, y)
        op = tf.add(xy, b, name='op_to_store')
        tf.io.write_graph(sess.graph, logdir="./", name="model.pb", as_text=False)

if __name__=='__main__':
    generate_conv2d_pb()
    generate_add_pb()
