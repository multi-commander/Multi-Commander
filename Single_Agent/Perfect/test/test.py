# import ray
# ray.init()


# @ray.remote(num_return_vals=3)
# def return_multiple():
#     return 1, 2, 3

# a_id, b_id, c_id = return_multiple.remote()
# print(ray.get([a_id, b_id]))

import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('log', sess.graph)
    i_p = tf.placeholder(tf.float32, [None, 1], name='i')
    
    tf.summary.scalar('i', i_p)
    merged = tf.summary.merge_all()

    for i in range(100):
        i_out, summary = sess.run([i_p, merged], feed_dict={i_p:np.reshape(np.array([i]), [-1, 1])})
        file_writer.add_summary(summary)


