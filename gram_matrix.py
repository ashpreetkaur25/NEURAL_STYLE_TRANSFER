"""importing the libraries"""
import tensorflow as tf


"""This file computes the style_matrix/Gram matrix of all the chosen style layers of the VGG-network
    which finds the correlation between the different filters of the layers"""


def gram(layer_filter):
    """"
    b - batch_size,
    i - height,
    j - width,
    c - channels in first matrix,
    d - same number of channels as of c but denoted with (d) as to create a gram/style matrix of
      shape (c x d) """

    result = tf.linalg.einsum('bijc,bijd->bcd', layer_filter, layer_filter)
    shape = tf.shape(layer_filter)
    i = shape[1]
    j = shape[2]
    num_locations = tf.cast(i * j, tf.float32)

    return result / num_locations

