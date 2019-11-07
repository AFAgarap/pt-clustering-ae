from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import tensorflow as tf


class CVAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CVAE, self).__init__()
        self.encoder = Encoder(
                input_shape=kwargs['input_shape'],
                latent_dim=kwargs['latent_dim']
                )
        self.decoder = Decoder(latent_dim=kwargs['latent_dim'])

    @tf.function
    def call(self, features):
        z_mean, z_log_var, z = self.encoder(features)
        reconstructed = self.decoder(z)
        kl_divergence = -5e-2 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var)
        self.add_loss(kl_divergence)
        return reconstructed


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
                input_shape=kwargs['input_shape']
                )
        self.conv_1_layer_1 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                activation=tf.nn.relu
                )
        self.conv_1_layer_2 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                activation=tf.nn.relu
                )
        self.conv_2_layer_1 = tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                activation=tf.nn.relu
                )
        self.conv_2_layer_2 = tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                activation=tf.nn.relu
                )
        self.flatten = tf.keras.layers.Flatten()
        self.z_mean_layer = tf.keras.layers.Dense(
                units=kwargs['latent_dim']
                )
        self.z_log_var_layer = tf.keras.layers.Dense(
                units=kwargs['latent_dim']
                )
        self.sampling = Sampling()

    def call(self, features):
        features = self.input_layer(features)
        activation = self.conv_1_layer_1(features)
        activation = self.conv_1_layer_2(activation)
        activation = self.conv_2_layer_1(activation)
        activation = self.conv_2_layer_2(activation)
        activation = self.flatten(activation)
        z_mean = self.z_mean_layer(activation)
        z_log_var = self.z_log_var_layer(activation)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
                input_shape=(kwargs['latent_dim'], )
                )
        self.hidden_layer_1 = tf.keras.layers.Dense(
                units=(28 * 28 * 32),
                activation=tf.nn.relu
                )
        self.reshape_layer = tf.keras.layers.Reshape(target_shape=(28, 28, 32))
        self.convt_1_layer_1 = tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu
                )
        self.convt_1_layer_2 = tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu
                )
        self.convt_2_layer_1 = tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu
                )
        self.convt_2_layer_2 = tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu
                )
        self.output_layer = tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                strides=(1, 1),
                padding='same',
                activation=tf.nn.sigmoid
                )

    def call(self, features):
        features = self.input_layer(features)
        activation = self.hidden_layer_1(features)
        activation = self.reshape_layer(activation)
        activation = self.convt_1_layer_1(activation)
        activation = self.convt_1_layer_2(activation)
        activation = self.convt_2_layer_1(activation)
        activation = self.convt_2_layer_2(activation)
        output = self.output_layer(activation)
        return output


class Sampling(tf.keras.layers.Layer):
    def call(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dimension = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dimension), mean=0., stddev=1.)
        return z_mean + epsilon + tf.exp(5e-1 * z_log_var)
