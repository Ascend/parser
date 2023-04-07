#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf
import numpy as np

def generate_VarIsInitializedOp_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=())
        y = tf.Variable(tf.compat.v1.random_normal(shape=[4,3],mean=0,stddev=1), dtype="float32", name='y')
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        op = tf.compat.v1.raw_ops.VarIsInitializedOp(resource=y, name="VarIsInitializedOp")
    tf.io.write_graph(sess.graph, logdir="./", name="test_VarIsInitializedOp.pb", as_text=False)

if __name__=='__main__':
    generate_VarIsInitializedOp_pb()
