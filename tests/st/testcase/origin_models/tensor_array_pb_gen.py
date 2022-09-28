#!/usr/bin/env python3
# -*- coding utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import gen_data_flow_ops
import numpy as np

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    size = tf.compat.v1.placeholder(dtype="int32", shape=())
    value = tf.compat.v1.placeholder(dtype="float32", shape=(2,2))
    index = tf.compat.v1.placeholder(dtype="int32", shape=())
    flow = tf.compat.v1.placeholder(dtype="float32", shape=())
    handleTensor = gen_data_flow_ops.tensor_array_v3(size= size, dtype = np.float32)
    output = gen_data_flow_ops.tensor_array_write_v3(handle = handleTensor[0], index=index, value=value, flow_in=flow)
    tf.io.write_graph(sess.graph, logdir="./", name="tensor_array.pb", as_text=False)
