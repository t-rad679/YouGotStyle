import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def fetch_file(name, url):
    """
    Pulls the image from a URL to a local cache directory and returns the path to the file

    :param name: (String) The filename of the resulting image
    :param url: (String) The URL for the file
    :return: (String) The path to the downloaded file
    """

    return tf.keras.utils.get_file(name, url)


def load_image_as_tensor(file_path):
    """
    Reads an image file from the disk and prepares it for use in the algorithm

    :param file_path: (String) Path to the file to open
    :return: (Tensor) A tensor representation of the image at the provided path
    """

    max_dim = 512
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image


def image_show(image, title=None):
    """
    Shows an image with pyplot or with image.show() if image is an instance of
    PIL.Image

    :param image: (Tensor) A Tensor representation of an image
    :param title: (String) A title to display on the plot
    """

    if type(image) is PIL.Image:
        image.show()
    new_image = image
    if len(image.shape) > 3:
        new_image = tf.squeeze(image, axis=0)
    plt.imshow(new_image)
    if title:
        plt.title(title)


def tensor_to_image(image):
    """
    Converts an image represented as a tensor to a PIL.Image object

    :param image: (Tensor) The tensor representation of the image to convert
    :return: (PIL.Image) The image in PIL.Image format
    """

    image = image * 255
    image = np.array(image, dtype=np.uint8)
    if (np.ndim(image)) > 3:
        assert image.shape[0] == 1
        image = image[0]
    return PIL.Image.fromarray(image)
