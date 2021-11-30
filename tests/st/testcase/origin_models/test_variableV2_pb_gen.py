import tensorflow as tf

def generate_VariableV2_pb():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = tf.compat.v1.placeholder(dtype="int32", shape=(2,3))
        op = tf.raw_ops.VariableV2(shape=[2,3], dtype="int32", name="VariableV2")
        init = tf.compat.v1.global_variables_initializer()
        op_add = tf.add(x,op)
        sess.run(init)
    tf.io.write_graph(sess.graph, logdir="./", name="test_VariableV2.pb", as_text=False)

if __name__=='__main__':
    generate_VariableV2_pb()