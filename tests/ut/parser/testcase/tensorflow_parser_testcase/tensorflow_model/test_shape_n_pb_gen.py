import tensorflow as tf
import numpy as np

def generate_shape_n_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
        y = tf.shape_n([1,2], name= "shape_n")
    tf.io.write_graph(sess.graph, logdir="./", name="test_shape_n.pb", as_text=False)

if __name__ == "__main__":
    generate_shape_n_pb()