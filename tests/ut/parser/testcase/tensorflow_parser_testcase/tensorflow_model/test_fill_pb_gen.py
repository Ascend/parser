import tensorflow as tf
import numpy as np

def generate_fill_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
        y = tf.fill([1,2], value = 5)
        z = tf.add(x,y)
    tf.io.write_graph(sess.graph, logdir="./", name="test_fill.pb", as_text=False)

if __name__ == "__main__":
    generate_fill_pb()