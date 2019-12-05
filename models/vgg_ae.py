"""Implementation of mini VGG-based autoencoder in TensorFlow 2.0"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "1.0.0"
__author__ = "Abien Fred Agarap"

import tensorflow as tf


class VGGAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(VGGAE, self).__init__()
        self.encoder = Encoder(input_shape=kwargs["input_shape"])
        self.decoder = Decoder(channels=kwargs["input_shape"][-1])

    def call(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=kwargs["input_shape"])
        self.conv_1_layer_1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.conv_1_layer_2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.conv_2_layer_1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.conv_2_layer_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation=tf.nn.sigmoid
        )

    def call(self, features):
        features = self.input_layer(features)
        activation = self.conv_1_layer_1(features)
        activation = self.conv_1_layer_2(activation)
        activation = self.conv_2_layer_1(activation)
        code = self.conv_2_layer_2(activation)
        return code


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.convt_1_layer_1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.convt_1_layer_2 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.convt_2_layer_1 = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.convt_2_layer_2 = tf.keras.layers.Conv2DTranspose(
            filters=kwargs["channels"],
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.sigmoid,
        )

    def call(self, features):
        activation = self.convt_1_layer_1(features)
        activation = self.convt_1_layer_2(activation)
        activation = self.convt_2_layer_1(activation)
        output = self.convt_2_layer_2(activation)
        return output
