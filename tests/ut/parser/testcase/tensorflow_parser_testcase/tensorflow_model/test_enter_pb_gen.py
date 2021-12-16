import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def generate_enter_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=())
        y = tf.compat.v1.placeholder(dtype="int32", shape=())
        output1 = control_flow_ops.enter(x, frame_name = "output1")
        output2 = control_flow_ops.enter(y, frame_name = "output2")
    tf.io.write_graph(sess.graph, logdir="./", name="test_enter.pb", as_text=False)

if __name__=='__main__':
    generate_enter_pb()