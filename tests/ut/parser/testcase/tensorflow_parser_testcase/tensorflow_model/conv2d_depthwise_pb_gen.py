#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf
import os

pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # NHWC 
    fmap_shape = [17, 101, 101, 17]
    filter_size = [5, 5, 17, 1]
    dy_shape = [17, 49, 49, 17]
    strideh, stridew = [2, 2]
    padding = 'VALID'
    tensor_x1 = tf.compat.v1.placeholder(dtype="float16", shape=fmap_shape)
    tensor_x2 = tf.compat.v1.placeholder(dtype="float16", shape=fmap_shape)
    tensor_x = tf.add(tensor_x1, tensor_x2)
    tensor_dy1 = tf.compat.v1.placeholder(dtype="float16", shape=dy_shape)
    tensor_dy2 = tf.compat.v1.placeholder(dtype="float16", shape=dy_shape)
    tensor_dy = tf.add(tensor_dy1, tensor_dy2)
    
    op = tf.nn.depthwise_conv2d_backprop_filter(tensor_x, filter_size, tensor_dy,
                                  strides=[1, strideh, stridew, 1],
                                  padding=padding,
                                  data_format='NHWC',
                                  dilations=[1,1,1,1])

    tf.io.write_graph(sess.graph, logdir="./", name="test_depth_wise_conv2d.pb", as_text=False)
