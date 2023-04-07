#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf
import numpy as np

def generate_no_op_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=())
        y = tf.compat.v1.placeholder(dtype="int32", shape=())
        add_op = tf.add(x, y)
        y2 = tf.no_op(name="train")
        z = tf.group([add_op, y2])
    tf.io.write_graph(sess.graph, logdir="./", name="test_no_op.pb", as_text=False)

if __name__=='__main__':
    generate_no_op_pb()
