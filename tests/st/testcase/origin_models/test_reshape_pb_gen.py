import tensorflow as tf

def generate_reshape_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
        y = tf.compat.v1.reshape(x, [2,2])
    tf.io.write_graph(sess.graph, logdir="./", name="test_reshape.pb", as_text=False)

if __name__ == "__main__":
    generate_reshape_pb()