import tensorflow as tf

state_shape = [80, 80, 4]
batch_size = 32
output_size = 3
activation = tf.nn.relu
final_activation = tf.nn.softmax
hidden = [256, 256, 256]
unroll = 5
entropy_coef = 0.01
reward_clip = ['tanh', 'abs_one', 'no_clip']

learner_ip = '0.0.0.0'
send_size = 32