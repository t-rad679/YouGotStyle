"""
Here be some functions for debugging changes to this project and testing out its
various components
"""
import os
from pathlib import Path
from re import search, match

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
    tf.keras.backend.clear_session()
    model = StyleContentModel(content_image, style_image, content_layers,
                              style_layers, model_type=model_type)
    image_var = tf.Variable(content_image)

    step = 0
    for i in range(epochs):
        for j in range(steps_per_epoch):
            step += 1
            st.train_step(model, image_var)
            print(".", end="")
        print("\n")
        print("Train step: {}".format(step))
    return tio.tensor_to_image(image_var)


def try_all_layer_combos(content_image, style_image,
                         epochs, steps_per_epoch,
                         project_path, image_name,
                         model_type=ModelType.VGG19):
    project_subdir = create_image_dir(project_path,
                                      "{}_combo".format(image_name))
    for current_content_layer in list_layers(model_type):
        content_subdir = create_image_dir(
                str(project_subdir) + os.path.sep,
                "{}_content".format(current_content_layer))
        try_all_style_layers(content_image, style_image,
                             epochs, steps_per_epoch,
                             str(content_subdir), image_name,
                             [current_content_layer],
                             model_type)


def try_all_content_layers(content_image, style_image,
                           epochs, steps_per_epoch,
                           project_path, image_name,
                           style_layers=ModelType.VGG19.content_layers,
                           model_type=ModelType.VGG19):
    image_dir = create_image_dir(project_path + os.path.sep,
                                 "{}_content".format(image_name))
    for current_content_layer in list_layers(model_type):
        final_image_name = image_name + "_" + current_content_layer + ".jpg"
        print(final_image_name)
        run_debug(content_image, style_image,
                  epochs, steps_per_epoch,
                  [current_content_layer], style_layers,
                  model_type).save(str(image_dir) +
                                   os.path.sep +
                                   final_image_name)


def try_all_style_layers(content_image, style_image,
                         epochs, steps_per_epoch,
                         project_path, image_name,
                         content_layers=ModelType.VGG19.content_layers,
                         model_type=ModelType.VGG19):
    image_dir = create_image_dir(project_path + os.path.sep,
                                 "{}_style".format(image_name))
    for current_style_layer in list_layers(model_type):
        final_image_name = image_name + "_" + current_style_layer + ".jpg"
        print(final_image_name)
        run_debug(content_image, style_image,
                  epochs, steps_per_epoch,
                  content_layers, [current_style_layer],
                  model_type).save(str(image_dir) + os.path.sep +
                                   final_image_name)


def create_image_dir(project_path, dir_name):
    new_image_dir = Path(project_path + dir_name)
    image_dir_num = 1
    while new_image_dir.exists():
        next_dir = Path(project_path +
                        dir_name + "_" + str(image_dir_num))
        if next_dir.exists():
            image_dir_num += 1
            continue
        else:
            new_image_dir = next_dir
    new_image_dir.mkdir()
    return new_image_dir


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
    vgg_layers = model_type.model_func(
            include_top=True, weights='imagenet').layers
    return [layer.name for layer in vgg_layers[1:-4]]


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
