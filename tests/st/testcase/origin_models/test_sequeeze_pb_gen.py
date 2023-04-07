#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf
import numpy as np

def generate_sequeeze_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
        y = tf.constant([[1,2],[2,3]])
        z = tf.add(x,y)
        op = tf.squeeze(z,name = "squeeze")
    tf.io.write_graph(sess.graph, logdir="./", name="test_sequeeze.pb", as_text=False)

if __name__ == "__main__":
    generate_sequeeze_pb()
