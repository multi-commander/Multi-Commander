import numpy as np
import tensorflow as tf

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def mlp(x, hidden, activation, output_size, final_activation):
    for h in hidden:
        x = tf.layers.dense(inputs=x, units=h, activation=activation)
    x = tf.layers.dense(inputs=x, units=output_size, activation=final_activation)
    return x

def cnn_model(x, hidden, activation, output_size, final_activation):
    x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    actor = mlp(x, hidden, activation, output_size, final_activation)
    value = tf.squeeze(mlp(x, hidden, activation, 1, None))
    return actor, value

def model(x, hidden, activation, output_size, final_activation):
    actor = mlp(x, hidden, activation, output_size, final_activation)
    value = tf.squeeze(mlp(x, hidden, activation, 1, None))
    return actor, value

def build_model(s, ns, hidden, activation, output_size, final_activation, state_shape, unroll, name):
    state = tf.reshape(s, [-1, *state_shape])
    next_state = tf.reshape(ns, [-1, *state_shape])

    if len(state_shape) == 1:
        with tf.variable_scope(name):
            policy, value = model(
                state, hidden, activation, output_size, final_activation
            )
        with tf.variable_scope(name, reuse=True):
            _, next_value = model(
                next_state, hidden, activation, output_size, final_activation
            )

    elif len(state_shape) == 3:
        with tf.variable_scope(name):
            policy, value = cnn_model(
                state, hidden, activation, output_size, final_activation
            )
        with tf.variable_scope(name, reuse=True):
            _, next_value = cnn_model(
                next_state, hidden, activation, output_size, final_activation
            )

    policy = tf.reshape(policy, [-1, unroll, output_size])
    value = tf.reshape(value, [-1, unroll])
    next_value = tf.reshape(next_value, [-1, unroll])

    return policy, value, next_value