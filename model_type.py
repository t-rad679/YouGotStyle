from enum import Enum, auto

import tensorflow as tf


class ModelData:
    """
    Stores functions and data representing a model
    """

    def __init__(self, model_func, model_stuff,
                 content_layers, style_layers):
        self.model_func = model_func
        self.model_stuff = model_stuff
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)


class ModelType(Enum):
    """
    All supported model types to be used in the style transfer algorithm and
    their associated data
    """

    def __init__(self, value):
        super().__init__()
        type_list = [ModelData(tf.keras.applications.VGG19,
                               tf.keras.applications.vgg19,
                               ['block5_conv2'],
                               ['block1_conv1',
                                'block2_conv1',
                                'block3_conv1',
                                'block4_conv1',
                                'block5_conv1']),

                     ModelData(tf.keras.applications.VGG16,
                               tf.keras.applications.vgg16,
                               ['block5_conv2'],
                               ['block1_conv1',
                                'block2_conv1',
                                'block3_conv1',
                                'block4_conv1',
                                'block5_conv1']),

                     # TODO: Add proper support for non-VGG models
                     # The issue primarily has to do with determining which
                     # layers to use as style and which to use as content.
                     # There is also the issue of robustness discussed here:
                     # https://distill.pub/2019/advex-bugs-discussion/response-4
                     # In short, non-VGG models are not naturally good for style
                     # transfer because they don't extract "robust features",
                     # while VGG models don't extract non-robust features.
                     ModelData(tf.keras.applications.ResNet50,
                               tf.keras.applications.resnet,
                               [], []),

                     ModelData(tf.keras.applications.InceptionV3,
                               tf.keras.applications.inception_v3,
                               [], []),

                     ModelData(tf.keras.applications.InceptionResNetV2,
                               tf.keras.applications.inception_resnet_v2,
                               [], []),

                     ModelData(tf.keras.applications.Xception,
                               tf.keras.applications.xception,
                               [], [])]
        self.model_func = type_list[value - 1].model_func
        # TODO: Come up with a better name for model_stuff
        self.model_stuff = type_list[value - 1].model_stuff
        self.content_layers = type_list[value - 1].content_layers
        self.style_layers = type_list[value - 1].style_layers
        self.num_content_layers = type_list[value - 1].num_content_layers
        self.num_style_layers = type_list[value - 1].num_style_layers

    VGG19 = auto()
    VGG16 = auto()
    RESNET = auto()
    INCEPTION_v3 = auto()
    INCEPTION_RESNET = auto()
    XCEPTION = auto()


