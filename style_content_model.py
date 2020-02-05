"""
Contains the custom model class used for extracting style and supporting data
types
"""

# Tensorflow imports
import tensorflow as tf

# Imports from this package
from const import STYLE_WEIGHT, CONTENT_WEIGHT
from maths import gram_matrix
import model_type as mt


class StyleContentModel(tf.keras.models.Model):
    """ Neural model of the style and content tensors combined. """

    def __init__(self,
                 content_image, style_image,
                 content_layers, style_layers,
                 model_type=mt.ModelType.VGG19,
                 weights_dataset='imagenet',
                 include_top=False):
        super(StyleContentModel, self).__init__()

        # Model fields
        self.model_type = model_type
        self.include_top = include_top
        self.weights_dataset = weights_dataset
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg = self.build_model(style_layers + content_layers, model_type)
        self.vgg.trainable = False

        # Training fields
        self.content_image = content_image
        self.style_image = style_image
        self.content_weight = CONTENT_WEIGHT
        self.style_weight = STYLE_WEIGHT

    def content_targets(self):
        """
        :return: (dict) The extracted data from the content layers
        """

        return self(self.content_image)['content']

    def style_targets(self):
        """
        :return: (dict) The extracted data from the style layers
        """

        return self(self.style_image)['style']

    def call(self, inputs, **kwargs):
        """
        Execute model.

        :param inputs: (Tensor) A tensor representing the style layers and
        the content layers
        :return: (dict) The Gram matrix of the style layers and content of the
        content layers
        """

        outputs = self.vgg(
                self.model_type.model_stuff.preprocess_input(inputs * 255))
        style_outputs = [gram_matrix(style_output)
                         for style_output
                         in outputs[:self.num_style_layers]]
        content_outputs = outputs[self.num_style_layers:]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    def build_model(self,
                    layer_names,
                    model_type):
        """
        Creates a VGG model that returns a list of intermediate output values.

        :param layer_names: (List<String>) A list of layer names
        :param model_type: (ModelType) The type of classification model to use
        :return (tf.keras.Model) A trained VGG model with the layers specified
        """

        vgg = model_type.model_func(include_top=self.include_top,
                                    weights=self.weights_dataset)
        vgg.trainable = False

        return tf.keras.Model([vgg.input],
                              [vgg.get_layer(name).output for name in
                               layer_names])

    def style_content_loss(self, outputs):
        """
        Calculates a weighted combination of optimization loss from the
        optimization in the training step

        :param outputs: (dict) A dictionary resulting from calling the Model
                               function
        :return: (Union<float, Any>) The weighted loss combination
        """
        content_outputs = outputs['content']
        style_outputs = outputs['style']

        content_loss = (tf.add_n([tf.reduce_mean(
                (content_outputs[name] - self.content_targets()[name]) ** 2)
                for name in content_outputs.keys()])
                        * (self.content_weight / self.num_content_layers))
        style_loss = (tf.add_n([tf.reduce_mean(
                (style_outputs[name] - self.style_targets()[name]) ** 2)
                for name in style_outputs.keys()])
                      * (self.style_weight / self.num_style_layers))
        return style_loss + content_loss


