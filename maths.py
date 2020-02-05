import tensorflow as tf


def gram_matrix(tensor):
    """
    Calculate Gram matrix (https://en.wikipedia.org/wiki/Gramian_matrix)

    :param tensor: (Tensor) The tensor for which to calculate the Gram matrix
    :return: The Gram matrix
    """
    input_shape = tf.shape(tensor)
    return tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor) / \
           (tf.cast(input_shape[1] * input_shape[2], tf.float32))