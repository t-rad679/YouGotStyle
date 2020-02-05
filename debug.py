"""
Here be some functions for debugging changes to this project and testing out its
various components
"""

import tensorflow as tf
import style_transfer as st
import tensor_io as tio
from model_type import ModelType
from style_content_model import StyleContentModel


def run_debug(content_image, style_image,
              epochs, steps_per_epoch,
              content_layers=ModelType.VGG19.content_layers,
              style_layers=ModelType.VGG19.style_layers,
              model_type=ModelType.VGG19):
    model = StyleContentModel(content_image, style_image, content_layers,
                              style_layers, model_type=model_type)
    image_var = tf.Variable(content_image)

    step = 0
    for i in range(epochs):
        for j in range(steps_per_epoch):
            step += 1
            st.train_step(model, image_var)
            print(".", end="")
        print("Train step: {}".format(step))
    return tio.tensor_to_image(image_var)


def test_style_content_model(content_image, style_image,
                             content_layers, style_layers):
    extractor = StyleContentModel(content_image, style_image,
                                     content_layers, style_layers)

    results = extractor(tf.constant(content_image))

    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("    ", name)
        print("        shape: ", output.numpy().shape)
        print("        min: ", output.numpy().min())
        print("        max: ", output.numpy().max())
        print("        mean: ", output.numpy().mean())
        print()
    print('content:')
    for name, output in sorted(results['content'].items()):
        print("    ", name)
        print("        shape: ", output.numpy().shape)
        print("        min: ", output.numpy().min())
        print("        max: ", output.numpy().max())
        print("        mean: ", output.numpy().mean())
        print()


def list_layers(model_type=ModelType.VGG19):
    vgg = model_type.model_func(include_top=False, weights='imagenet')

    print()
    for layer in vgg.layers:
        print(layer.name)


def layer_info(layers, outputs):
    for name, output in zip(layers, outputs):
        print(name)
        print("    shape ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()


def define_representation(image, include_top=True):
    """
    Extracts a feature representation of the image using a pre-trained network
    :param image: (Tensor) A tensor representation of the image
    :param include_top: (Boolean) Whether to include the top three layers of
    the VGG19 network

    :return: ???
    """
    preprocess = tf.keras.applications.vgg19.preprocess_input(image * 255)
    preprocess = tf.image.resize(preprocess, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=include_top,
                                      weights='imagenet')
    prediction_probabilities = vgg(preprocess)

    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(
            prediction_probabilities.numpy())[0]
    return [(class_name, prob) for (number, class_name, prob) in
            predicted_top_5]
    # return prediction_probabilities.shape
