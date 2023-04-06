#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf

def generate_identity_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=())
        x_plus_1 = tf.add(x, 1, name='x_plus')
        with tf.control_dependencies([x_plus_1]):
            y = x
            z = tf.identity(x,name='identity')
    tf.io.write_graph(sess.graph, logdir="./", name="test_identity.pb", as_text=False)

if __name__=='__main__':
    generate_identity_pb()
