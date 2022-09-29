#!/usr/bin/env python3
# -*- coding utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def generate_switch_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=())
        y = tf.compat.v1.placeholder(dtype="int32", shape=())
        output1 = control_flow_ops.switch(x,False)
        output2 = control_flow_ops.switch(y,True)
    tf.io.write_graph(sess.graph, logdir="./", name="test_switch.pb", as_text=False)

if __name__=='__main__':
    generate_switch_pb()
