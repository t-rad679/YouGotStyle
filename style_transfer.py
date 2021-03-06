"""
Functions to perform style transfer
"""

# stdlib imports
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)

# Tensorflow imports
import tensorflow as tf
import tensorflow_hub as hub

# Imports from this package
from const import BETA_1, TOTAL_VARIATION_WEIGHT, LEARNING_RATE, EPSILON


# Environment setup
tf.config.experimental_run_functions_eagerly(True)


def style_transfer_easy_mode(content, style):
    """
    Uses the Fast Style Transfer project on TF Hub to perform the style transfer

    :param content: (Tensor) A tensor representation of the content image
    :param style: (Tensor) A tensor representation of the style image
    :return: (PIL.Image) The newly stylized image
    """
    hub_module = hub.load(
            'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1'
            '-256/1')
    return hub_module(tf.constant(content), tf.constant(style))[0]

    # TODO: The return type seems to be inconsistent. Figure that shit out.
    # return tio.tensor_to_image(stylized_image)


def clip_0_1(image):
    """
    clip values of the tensor to between 0 and 1

    :param image: (Tensor) A Tensor representation of an image
    :return: (tf.Tensor) The Tensor of the image clipped to between 0 and 1
    """

    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


@tf.function()
def train_step(extractor,
               image_var,
               total_variation_weight=TOTAL_VARIATION_WEIGHT,
               optimizer_type=tf.optimizers.Adam):
    """
    Performs the training of the model that extracts the style

    :param extractor: (StyleContentModel) The model to perform training
    :param image_var: (tf.Variable) A Variable containing the image Tensor
    :param total_variation_weight: (Integer) A constant modifier for the
                                             total variation of the image
    :param optimizer_type: (tf.keras.optimizer_v2.OptimizerV2) The optimization
                            function used to train the model
    :return: (tf.Variable) A Variable containing the trained Tensor
    """

    with tf.GradientTape() as tape:
        outputs = extractor(image_var)
        loss = extractor.style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image_var)
    grad = tape.gradient(loss, image_var)
    opt = optimizer_type(learning_rate=LEARNING_RATE,
                         beta_1=BETA_1,
                         epsilon=EPSILON)
    opt.apply_gradients([(grad, image_var)])
    image_var.assign(clip_0_1(image_var))
