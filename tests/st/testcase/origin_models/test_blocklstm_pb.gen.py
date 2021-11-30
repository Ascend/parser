import tensorflow as tf
import os

pb_file_path = os.getcwd()

def generate_case_0():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        input_dtype = tf.float32
        input_shape0 = [1, ]
        input_shape1 = [202, 1, 768]
        input_shape2 = [1, 1]
        input_shape3 = [1, 1]
        input_shape4 = [769, 4]
        input_shape5 = [1, ]
        input_shape6 = [1, ]
        input_shape7 = [1, ]
        input_shape8 = [4, ]

        d0 = tf.compat.v1.placeholder(dtype=tf.int64, shape=input_shape0)
        d1 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape1)
        d2 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape2)
        d3 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape3)
        d4 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape4)
        d5 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape5)
        d6 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape6)
        d7 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape7)
        d8 = tf.compat.v1.placeholder(dtype=input_dtype, shape=input_shape8)

        i1, cs1, f1, o1, ci1, co1, h1 = tf.raw_ops.BlockLSTM(seq_len_max=d0, x=d1, cs_prev=d2, h_prev=d3, w=d4, wci=d5, wcf=d6, wco=d7, b=d8,
                                                             forget_bias=1, cell_clip=3, use_peephole=False, name="blockLSTM")
        tf.io.write_graph(sess.graph, logdir="./", name="blocklstm_case.pb", as_text=False)

if __name__=='__main__':
    generate_case_0()
