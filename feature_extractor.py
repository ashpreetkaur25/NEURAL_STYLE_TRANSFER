"""Importing the libraries"""
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from gram_matrix import gram

"""defining the own vgg-model for the desired output"""


def new_model(model_name, layers):
    """extracting the necessary layers from VGG model"""
    layer_output = [model_name.get_layer(layer_name).output for layer_name in layers]

    """Developing the model to get the desired results from specified style and content layers"""
    new_vgg = Model([model_name.input], layer_output)

    return new_vgg


class FeatureExtractor(tf.keras.models.Model):
    def __init__(self, content_layers, style_layers, model):
        super(FeatureExtractor, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg = new_model(model, content_layers + style_layers)

    def features(self, inputs):
        """Rescaling the image pixel values"""
        inputs = inputs * 255.0
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        """extracting the content and style features"""
        content_feature = list(outputs[:self.num_content_layers])
        style_features = list(outputs[self.num_content_layers:])

        """extracting the gram matrix out of the layers"""
        gram_matrices = [
            gram(layer_features) for layer_features in style_features
        ]

        """content features dictionary"""
        content_dict = {
            layer_name: value
            for layer_name, value in zip(self.content_layers, content_feature)
        }

        """style features dictionary"""
        style_dict = {
            layer_name: value
            for layer_name, value in zip(self.style_layers, gram_matrices)
        }

        return {
            "content": content_dict,
            "style": style_dict,
        }

